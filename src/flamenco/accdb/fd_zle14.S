/* fd_zle14: a zero run length codec in x86-64 assembly.  See fd_zle14.h
   for the API.

   WIRE FORMAT

   The input is cut into blocks of 64 bytes; the last block holds the
   remaining 1..64.  Blocks are grouped into chunks, and a chunk keeps its
   control bytes and its literal bytes in two runs rather than
   interleaved, so that neither side has to walk a variable length field
   to find where the next one starts:

     blob   := chunk+                       (empty input -> empty blob)
     chunk  := hdr[2] ctl[ctl_sz] lit[...]
     hdr    := u16le: ctl_sz<<7 | (rem-1)<<1 | last
     ctl    := group+                       (covers this chunk's blocks)
     lit    := the literal bytes of this chunk's blocks, in order

   rem is the size of the blob's very last block (1..64) and last is set
   on the chunk that holds it.  A chunk takes blocks until its control
   field would not hold another span of them (ctl_sz reaches 200, and a
   span adds at most 288) or its 2048 byte literal run is full, so
   ctl_sz<=488 and the three fields fit one u16.

   A group describes one block, or a run of trivial blocks.  Its first
   byte c selects the form:

     c&0x40==0  position list: c and the bytes that follow are byte
                positions inside the block, bit 7 set on the last one.
                Positions are strictly increasing and alternate: the
                first starts a zero run, the second ends it, and so on.
                The block's literals are the bytes outside those runs,
                read from lit in order.  At most 4 positions (below).
     c==0x7F    mask mode: the next 8 bytes are a little endian bit mask
                of the block's literal bytes.
     0x40..0x7E (c-0x3F) blocks, 1..63, that are all literal.
     0xC0..0xFF (c-0xBF) blocks, 1..64, that are all zero.

   A zero run is coded only if it is at least 2 bytes long and lies
   entirely inside its block; single zeros, and the 1 byte tails of runs
   that straddle a block boundary, stay literals.  The coder switches to
   mask mode once a block has 5 transitions, so a position list is never
   longer than 4 bytes and a group never longer than 9.  The mask costs 9
   bytes against a list's 5 to 8 -- 0.2% of the output -- and buys a
   group whose length neither side has to find by scanning.

   The block is the unit of work on both sides: the encoder advances the
   input by 64 bytes per block and the decoder advances the output by 64
   bytes per block, while the other cursor advances by a popcount of the
   block's mask, never by a length read out of the stream.  That is where
   the speed comes from: the dense shapes carry no serial chain through
   the data at all. */

#if defined(__x86_64__) && defined(FD_HAS_AVX512)

/* Make the stack non-executable */
.section .note.GNU-stack,"",@progbits

	.section .rodata
	.align	64
.Lzle14_iota:
	.byte	 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15
	.byte	16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
	.byte	32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
	.byte	48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
	.align	8
.Lzle14_pos:					/* clamp position bytes to 0..63 */
	.quad	0x3F3F3F3F3F3F3F3F

/* ulong fd_zle14_compress( void * comp, void const * data, ulong data_sz )

   rdi out (control) cursor   rsi src cursor      rdx literal cursor
   r9  src_end                r8  chunk src end   rax rcx r10 r11 scratch
   0(%rsp) dst base           8(%rsp) chunk ctl start
   16(%rsp) last chunk hdr    64(%rsp) literal scratch
   zmm31 iota                 zmm0 block data

   Everything lives in call clobbered registers, so the function saves
   nothing.  r8 is the last block boundary of the chunk, so a block is
   full whenever src<r8 and the block loop needs no size test. */

#define ZLE14_LIT   64			/* literal scratch, off rsp */
#define ZLE14_SPAN  2048		/* literal bytes a chunk can hold */
#define ZLE14_FRAME 2240

	.text
	.globl	fd_zle14_compress
	.type	fd_zle14_compress,@function
	.align	64
fd_zle14_compress:
	.cfi_startproc
	endbr64
	testq	%rdx, %rdx
	jz	.Lc_empty
	subq	$ZLE14_FRAME, %rsp
	.cfi_adjust_cfa_offset ZLE14_FRAME
	movq	%rdi, (%rsp)			/* dst base */
	leaq	(%rsi,%rdx), %r9		/* src_end */
	leaq	-1(%rdx), %rax
	andl	$63, %eax
	addq	%rax, %rax
	orq	$1, %rax
	movq	%rax, 16(%rsp)			/* (rem-1)<<1 | 1 */
	vmovdqu64 .Lzle14_iota(%rip), %zmm31

.Lc_chunk:
	addq	$2, %rdi			/* the chunk header */
	movq	%rdi, 8(%rsp)
	leaq	ZLE14_LIT(%rsp), %rdx
	movq	%r9, %rax			/* a fresh chunk: all scratch free */
	subq	%rsi, %rax
	andq	$-64, %rax			/* whole blocks left */
	cmpq	$ZLE14_SPAN, %rax
	jbe	.Lc_span_short
	movl	$ZLE14_SPAN, %eax
	jmp	.Lc_span_short
.Lc_span:					/* a later span of the same chunk */
	movq	%r9, %rax
	subq	%rsi, %rax
	andq	$-64, %rax			/* whole blocks left */
	leaq	ZLE14_LIT+ZLE14_SPAN(%rsp), %rcx
	subq	%rdx, %rcx			/* literal scratch still free */
	cmpq	%rcx, %rax
	jbe	.Lc_span_short
	movq	%rcx, %rax
	andq	$-64, %rax
.Lc_span_short:
	leaq	(%rsi,%rax), %r8
	cmpq	%r8, %rsi
	jae	.Lc_tail			/* no full block left */

	.align	32
.Lc_blk:					/* a full block at src */
	vmovdqu64 (%rsi), %zmm0
	vptestnmb %zmm0, %zmm0, %k1		/* zero bytes */
	kmovq	%k1, %rax
	movq	%rax, %r10
	shrq	$1, %r10
	andq	%rax, %r10			/* r2: 2 zeros start here */
	jz	.Lc_alllit
.Lc_blk_r2:					/* r10 r2 nonzero, zmm0 data */
	leaq	(%r10,%r10), %r11
	orq	%r11, %r10			/* b: byte is in a coded run */
	cmpq	$-1, %r10
	je	.Lc_allzero
	leaq	(%r10,%r10), %r11
	xorq	%r10, %r11			/* T: transitions */
	notq	%r10				/* keep */
	popcntq	%r11, %rcx			/* t */
	cmpq	$5, %rcx
	jae	.Lc_mask
	cmpq	$1, %rcx
	je	.Lc_one
	kmovq	%r11, %k2
	vpcompressb %zmm31, (%rdi){%k2}		/* the positions */
	lzcntq	%r11, %rax
	xorq	$63, %rax
	orq	$0x80, %rax
	movb	%al, -1(%rdi,%rcx,1)		/* terminate the list */
	addq	%rcx, %rdi
.Lc_lits:					/* r10 keep, zmm0 data */
	kmovq	%r10, %k3
	popcntq	%r10, %rax
	vpcompressb %zmm0, (%rdx){%k3}
	addq	%rax, %rdx
	addq	$64, %rsi
	cmpq	%r8, %rsi
	jb	.Lc_blk

.Lc_blkdone:					/* the chunk's span is done */
	cmpq	%r9, %rsi
	jae	.Lc_chunk_end
	movq	%r9, %rax
	subq	%rsi, %rax
	cmpq	$64, %rax
	jb	.Lc_tail			/* a short last block, same chunk */
	movq	%rdi, %rax
	subq	8(%rsp), %rax
	cmpq	$200, %rax			/* a span adds at most 288 bytes */
	jae	.Lc_chunk_end			/* the control field would not hold it */
	leaq	ZLE14_LIT+ZLE14_SPAN-63(%rsp), %rax
	cmpq	%rax, %rdx
	jae	.Lc_chunk_end			/* the literal scratch is full */
	jmp	.Lc_span			/* same chunk, next span */

/* One zero run, reaching the end of the block: one control byte, and the
   literals are the block's own prefix, so neither needs a compress. */
	.align	16
.Lc_one:
	tzcntq	%r11, %rcx
	movl	%ecx, %eax
	orl	$0x80, %eax
	movb	%al, (%rdi)
	incq	%rdi
	vmovdqu64 %zmm0, (%rdx)
	addq	%rcx, %rdx
	addq	$64, %rsi
	cmpq	%r8, %rsi
	jb	.Lc_blk
	jmp	.Lc_blkdone

	.align	16
.Lc_mask:
	movb	$0x7F, (%rdi)
	movq	%r10, 1(%rdi)
	addq	$9, %rdi
	jmp	.Lc_lits

/* A block with no coded zero run.  Runs of them share one group, and
   the loop's test doubles as the next block's, so falling out of it
   re-enters the block body with the mask already in hand. */
	.align	16
.Lc_alllit:
	xorl	%ecx, %ecx
.Lc_al_loop:
	vmovdqu64 %zmm0, (%rdx)
	addq	$64, %rdx
	addq	$64, %rsi
	incl	%ecx
	cmpq	%r8, %rsi
	jae	.Lc_al_last
	vmovdqu64 (%rsi), %zmm0
	vptestnmb %zmm0, %zmm0, %k1
	kmovq	%k1, %rax
	movq	%rax, %r10
	shrq	$1, %r10
	andq	%rax, %r10
	jz	.Lc_al_loop
	leal	63(%rcx), %eax
	movb	%al, (%rdi)
	incq	%rdi
	jmp	.Lc_blk_r2
.Lc_al_last:
	leal	63(%rcx), %eax
	movb	%al, (%rdi)
	incq	%rdi
	jmp	.Lc_blkdone

/* An all zero block.  Scan 256 bytes at a time for the end of the run,
   out to the 64 blocks one group can code, regardless of where the
   chunk's span ends -- the run costs the chunk no literal scratch. */
	.align	16
.Lc_allzero:
	movq	%rsi, %r11			/* run start */
	movq	%r9, %r10
	subq	%rsi, %r10
	andq	$-64, %r10			/* whole blocks left */
	cmpq	$4096, %r10
	jbe	.Lc_az_lim
	movl	$4096, %r10d			/* a group codes 64 blocks at most */
.Lc_az_lim:
	addq	%rsi, %r10			/* scan limit */
	addq	$64, %rsi
.Lc_az_wide:
	leaq	256(%rsi), %rcx
	cmpq	%r10, %rcx
	ja	.Lc_az_one
	vmovdqu64 (%rsi), %zmm1
	vpord	64(%rsi), %zmm1, %zmm1
	vpord	128(%rsi), %zmm1, %zmm1
	vpord	192(%rsi), %zmm1, %zmm1
	vptestmb %zmm1, %zmm1, %k1
	kortestq %k1, %k1
	jnz	.Lc_az_one
	movq	%rcx, %rsi
	jmp	.Lc_az_wide
.Lc_az_one:
	cmpq	%r10, %rsi
	jae	.Lc_az_emit
	vmovdqu64 (%rsi), %zmm1
	vptestmb %zmm1, %zmm1, %k1
	kortestq %k1, %k1
	jnz	.Lc_az_emit
	addq	$64, %rsi
	jmp	.Lc_az_one
.Lc_az_emit:
	movq	%rsi, %rax
	subq	%r11, %rax
	shrq	$6, %rax			/* blocks in the run */
	addl	$191, %eax
	movb	%al, (%rdi)
	incq	%rdi
	cmpq	%r8, %rsi
	jb	.Lc_blk
	jmp	.Lc_blkdone

/* The last block, 1..63 bytes.  All zero is tested first: an account
   that ends in zeros is the common case.  The load reads up to
   align_up( data_sz, 64 ). */
	.align	16
.Lc_tail:
	movq	%r9, %rcx
	subq	%rsi, %rcx			/* block size */
	movq	$-1, %rax
	bzhi	%rcx, %rax, %rcx		/* full */
	vmovdqu64 (%rsi), %zmm0
	vptestmb %zmm0, %zmm0, %k1		/* nonzero bytes */
	kmovq	%k1, %rax
	movq	%r9, %rsi
	testq	%rcx, %rax
	jz	.Lc_t_allzero
	andn	%rcx, %rax, %rax		/* z = ~m & full */
	movq	%rax, %r10
	shrq	$1, %r10
	andq	%rax, %r10			/* r2 */
	jz	.Lc_t_alllit
	leaq	(%r10,%r10), %r11
	orq	%r11, %r10
	andq	%rcx, %r10			/* b */
	leaq	(%r10,%r10), %r11
	xorq	%r10, %r11
	andq	%rcx, %r11			/* T */
	andn	%rcx, %r10, %r10		/* keep = ~b & full */
	popcntq	%r11, %rcx
	cmpq	$5, %rcx
	jae	.Lc_t_mask
	kmovq	%r11, %k2
	vpcompressb %zmm31, (%rdi){%k2}
	lzcntq	%r11, %rax
	xorq	$63, %rax
	orq	$0x80, %rax
	movb	%al, -1(%rdi,%rcx,1)
	addq	%rcx, %rdi
.Lc_t_lits:
	kmovq	%r10, %k3
	popcntq	%r10, %rax
	vpcompressb %zmm0, (%rdx){%k3}
	addq	%rax, %rdx
	jmp	.Lc_chunk_end
.Lc_t_mask:
	movb	$0x7F, (%rdi)
	movq	%r10, 1(%rdi)
	addq	$9, %rdi
	jmp	.Lc_t_lits
.Lc_t_alllit:
	movb	$0x40, (%rdi)
	incq	%rdi
	kmovq	%rcx, %k3
	vpcompressb %zmm0, (%rdx){%k3}		/* the whole block */
	popcntq	%rcx, %rax
	addq	%rax, %rdx
	jmp	.Lc_chunk_end
.Lc_t_allzero:
	movb	$0xC0, (%rdi)
	incq	%rdi

.Lc_chunk_end:
	movq	8(%rsp), %r10			/* chunk ctl start */
	movq	%rdi, %rax
	subq	%r10, %rax
	shlq	$7, %rax			/* ctl_sz<<7 */
	movl	$126, %ecx			/* rem 64, last 0 */
	cmpq	%r9, %rsi
	cmovaeq	16(%rsp), %rcx
	orq	%rcx, %rax
	movw	%ax, -2(%r10)
	leaq	ZLE14_LIT(%rsp), %r10
	movq	%rdx, %rcx
	subq	%r10, %rcx			/* lit_sz */
	jz	.Lc_copied
	cmpq	$64, %rcx
	ja	.Lc_copy
	vmovdqu64 (%r10), %zmm1			/* one block covers most chunks */
	vmovdqu64 %zmm1, (%rdi)
	jmp	.Lc_copied
.Lc_copy:
	xorl	%eax, %eax
.Lc_copy_loop:
	vmovdqu64 (%r10,%rax), %zmm1
	vmovdqu64 %zmm1, (%rdi,%rax)
	addq	$64, %rax
	cmpq	%rcx, %rax
	jb	.Lc_copy_loop
.Lc_copied:
	addq	%rcx, %rdi
	cmpq	%r9, %rsi
	jb	.Lc_chunk

	movq	%rdi, %rax
	subq	(%rsp), %rax
	vzeroupper
	addq	$ZLE14_FRAME, %rsp
	.cfi_adjust_cfa_offset -ZLE14_FRAME
	ret
.Lc_empty:
	xorl	%eax, %eax
	ret
	.cfi_endproc
	.size	fd_zle14_compress,.-fd_zle14_compress

/* long fd_zle14_decompress( void * data, ulong data_sz,
                             void const * comp, ulong comp_sz )

   rdi out cursor             r9  out_end           r8  blob_end
   rbx control cursor         rbp control end       r12 literal cursor
   r14 out_end-128            rsi 0x8080..80        zmm31 zero
   rax rcx rdx r10 r11        scratch
   0(%rsp) data base          24(%rsp) last block size
   32(%rsp) literal read lim  64(%rsp) padded control copy

   A group is emitted with sloppy 64 byte stores when it is not the last
   group of its chunk and both cursors have a block of slack.  Otherwise
   it goes through the mask path, which expands the block into one
   register and stores exactly its bytes, so nothing is ever read past
   comp+comp_sz or written past data+data_sz. */

#define ZLE14_DCTL   64			/* ctl scratch, off rsp */
#define ZLE14_DFRAME 704		/* 64 + ctl_sz's 511 max, rounded up by a
					   block: the padded copy stores 64 at a
					   time */

	.text
	.globl	fd_zle14_decompress
	.type	fd_zle14_decompress,@function
	.align	64
fd_zle14_decompress:
	.cfi_startproc
	endbr64
	testq	%rcx, %rcx
	jz	.Ld_zero
	pushq	%rbx
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	subq	$ZLE14_DFRAME, %rsp
	.cfi_adjust_cfa_offset ZLE14_DFRAME

	movq	%rdi, (%rsp)			/* data base */
	leaq	(%rdi,%rsi), %r9		/* out_end */
	leaq	(%rdx,%rcx), %r8		/* blob_end */
	leaq	-128(%r9), %r14
	movabsq	$0x8080808080808080, %rsi
	vpxorq	%zmm31, %zmm31, %zmm31
	/* The sloppy paths read up to 128 bytes past the literal cursor.
	   When the blob's last byte is at least that far from the end of its
	   page, those reads touch no page the blob does not already touch,
	   and the cursor only has to stay inside the blob. */
	leaq	-1(%r8), %rax
	andl	$4095, %eax
	leaq	-128(%r8), %rcx
	movq	%r8, %r10
	cmpl	$3968, %eax
	cmovaeq	%rcx, %r10
	movq	%r10, 32(%rsp)

.Ld_chunk:					/* rdx < blob_end */
	leaq	2(%rdx), %rbx			/* control */
	cmpq	%r8, %rbx
	ja	.Ld_estream
	movzwl	(%rdx), %eax
	movl	%eax, %ecx
	shrl	$7, %ecx			/* ctl_sz */
	jz	.Ld_estream
	leaq	(%rbx,%rcx), %r12		/* literals */
	cmpq	%r8, %r12
	ja	.Ld_estream
	movq	%r12, %rbp			/* control end */
	movl	$64, %ecx
	testb	$1, %al
	jz	.Ld_chunk_rem
	shrl	$1, %eax
	andl	$63, %eax
	leal	1(%rax), %ecx			/* the blob's last block */
.Ld_chunk_rem:
	movq	%rcx, 24(%rsp)
.Ld_chunk_go:
	leaq	-120(%rbp), %rax		/* the 8 byte control reads must fit */
	cmpq	32(%rsp), %rax
	ja	.Ld_ctlcopy
	movq	(%rbx), %rax

	.align	32
.Ld_grp:					/* rbx < rbp, rax = control bytes at rbx */
	testb	$0x40, %al
	jnz	.Ld_special
	movq	%rax, %r10
	andq	%rsi, %r10
	tzcntq	%r10, %r10
	shrl	$3, %r10d			/* t-1 */
	andq	.Lzle14_pos(%rip), %rax		/* a corrupt stream can set bit 6 */
	leaq	1(%rbx,%r10), %rcx
	cmpq	%rbp, %rcx
	jae	.Ld_pos_exact			/* the chunk's last group */
	cmpq	%r14, %rdi
	ja	.Ld_pos_exact			/* output slack */
	cmpq	32(%rsp), %r12
	ja	.Ld_pos_exact			/* literal slack */
	movq	%rcx, %rbx
	cmpl	$2, %r10d			/* two zero runs is the common case */
	je	.Ld_t3
	jb	.Ld_t12
	cmpl	$3, %r10d
	je	.Ld_t4
	jmp	.Ld_pos_exact5
.Ld_t12:
	testl	%r10d, %r10d
	jz	.Ld_t1
	jmp	.Ld_t2

/* One zero run reaching the end of the block. */
	.align	16
.Ld_t1:
	movzbl	%al, %ecx			/* p0 */
	movq	(%rbx), %rax			/* the next group, early */
	vmovdqu64 (%r12), %zmm0
	vmovdqu64 %zmm0, (%rdi)
	vmovdqu64 %zmm31, (%rdi,%rcx,1)
	addq	%rcx, %r12
	addq	$64, %rdi
	jmp	.Ld_grp

/* One zero run inside the block. */
	.align	16
.Ld_t2:
	movzbl	%al, %ecx			/* p0 */
	shrq	$8, %rax
	movzbl	%al, %r10d			/* p1 */
	movq	(%rbx), %rax			/* the next group, early */
	vmovdqu64 (%r12), %zmm0
	vmovdqu64 (%r12,%rcx,1), %zmm1
	vmovdqu64 %zmm0, (%rdi)
	vmovdqu64 %zmm31, (%rdi,%rcx,1)
	vmovdqu64 %zmm1, (%rdi,%r10,1)
	subq	%r10, %rcx
	addq	$64, %rcx			/* p0 + 64 - p1 */
	addq	%rcx, %r12
	addq	$64, %rdi
	jmp	.Ld_grp

/* Two zero runs, the second reaching the end of the block. */
	.align	16
.Ld_t3:
	movzbl	%al, %ecx			/* p0 */
	shrq	$8, %rax
	movzbl	%al, %r10d			/* p1 */
	shrq	$8, %rax
	movzbl	%al, %r11d			/* p2 */
	movq	(%rbx), %rax			/* the next group, early */
	vmovdqu64 (%r12), %zmm0
	vmovdqu64 (%r12,%rcx,1), %zmm1
	vmovdqu64 %zmm0, (%rdi)
	vmovdqu64 %zmm31, (%rdi,%rcx,1)
	vmovdqu64 %zmm1, (%rdi,%r10,1)
	vmovdqu64 %zmm31, (%rdi,%r11,1)
	subq	%r10, %r11			/* p2-p1 */
	addq	%r11, %rcx			/* p0+p2-p1 */
	addq	%rcx, %r12
	addq	$64, %rdi
	jmp	.Ld_grp

/* Two zero runs inside the block. */
	.align	16
.Ld_t4:
	movzbl	%al, %ecx			/* p0 */
	shrq	$8, %rax
	movzbl	%al, %r10d			/* p1 */
	shrq	$8, %rax
	movzbl	%al, %r11d			/* p2 */
	shrq	$8, %rax
	movzbl	%al, %edx			/* p3 */
	movq	(%rbx), %rax			/* the next group, early */
	vmovdqu64 (%r12), %zmm0
	vmovdqu64 (%r12,%rcx,1), %zmm1
	vmovdqu64 %zmm0, (%rdi)
	vmovdqu64 %zmm31, (%rdi,%rcx,1)
	vmovdqu64 %zmm1, (%rdi,%r10,1)
	vmovdqu64 %zmm31, (%rdi,%r11,1)
	subq	%r10, %r11
	addq	%r11, %rcx			/* p0+p2-p1 */
	andl	$63, %ecx			/* a corrupt stream can wrap it */
	vmovdqu64 (%r12,%rcx,1), %zmm2
	vmovdqu64 %zmm2, (%rdi,%rdx,1)
	addq	$64, %rcx
	subq	%rdx, %rcx			/* + 64-p3 */
	addq	%rcx, %r12
	addq	$64, %rdi
	jmp	.Ld_grp

/* Special groups.  One all zero block and one all literal block are
   half of all groups, so they get their own short paths. */
	.align	16
.Ld_special:
	cmpb	$0xC0, %al
	je	.Ld_z1
	ja	.Ld_zrun
	cmpb	$0x40, %al
	je	.Ld_l1
	cmpb	$0x7F, %al
	je	.Ld_mask
.Ld_lrun:
	movzbl	%al, %ecx
	subl	$0x3F, %ecx
	shlq	$6, %rcx			/* literal bytes */
	leaq	1(%rbx), %rax
	cmpq	%rbp, %rax
	jae	.Ld_lrun_last
	movq	%rax, %rbx
.Ld_lrun_go:
	leaq	(%rdi,%rcx), %rax
	cmpq	%r9, %rax
	ja	.Ld_espace
	leaq	(%r12,%rcx), %rax
	cmpq	%r8, %rax
	ja	.Ld_estream
	movq	%rcx, %rdx
	andq	$-64, %rdx
	xorl	%eax, %eax
	cmpq	$256, %rdx
	jb	.Ld_lrun_one
.Ld_lrun_four:
	vmovdqu64 (%r12,%rax), %zmm0
	vmovdqu64 64(%r12,%rax), %zmm1
	vmovdqu64 128(%r12,%rax), %zmm2
	vmovdqu64 192(%r12,%rax), %zmm3
	vmovdqu64 %zmm0, (%rdi,%rax)
	vmovdqu64 %zmm1, 64(%rdi,%rax)
	vmovdqu64 %zmm2, 128(%rdi,%rax)
	vmovdqu64 %zmm3, 192(%rdi,%rax)
	addq	$256, %rax
	leaq	256(%rax), %r10
	cmpq	%rdx, %r10
	jbe	.Ld_lrun_four
.Ld_lrun_one:
	cmpq	%rdx, %rax
	jae	.Ld_lrun_tail
	vmovdqu64 (%r12,%rax), %zmm0
	vmovdqu64 %zmm0, (%rdi,%rax)
	addq	$64, %rax
	jmp	.Ld_lrun_one
.Ld_lrun_tail:
	movq	%rcx, %rax
	andl	$63, %eax
	jz	.Ld_lrun_done
	movq	$-1, %r10
	bzhi	%rax, %r10, %r10
	kmovq	%r10, %k1
	vmovdqu8 (%r12,%rdx), %zmm0{%k1}{z}
	vmovdqu8 %zmm0, (%rdi,%rdx){%k1}
.Ld_lrun_done:
	addq	%rcx, %r12
	addq	%rcx, %rdi
	cmpq	%rbp, %rbx
	jae	.Ld_chunk_next
	movq	(%rbx), %rax
	jmp	.Ld_grp
.Ld_lrun_last:					/* the blob's last block may be short */
	movq	%rax, %rbx
	subq	$64, %rcx
	addq	24(%rsp), %rcx
	jmp	.Ld_lrun_go

	.align	16
.Ld_z1:						/* one all zero block */
	leaq	1(%rbx), %rcx
	cmpq	%rbp, %rcx
	jae	.Ld_z1_last
	cmpq	%r14, %rdi
	ja	.Ld_zrun
	movq	%rcx, %rbx
	movq	(%rbx), %rax
	vmovdqu64 %zmm31, (%rdi)
	addq	$64, %rdi
	jmp	.Ld_grp

	.align	16
.Ld_l1:						/* one all literal block */
	leaq	1(%rbx), %rcx
	cmpq	%rbp, %rcx
	jae	.Ld_l1_last
	cmpq	%r14, %rdi
	ja	.Ld_lrun
	cmpq	32(%rsp), %r12
	ja	.Ld_lrun
	movq	%rcx, %rbx
	movq	(%rbx), %rax
	vmovdqu64 (%r12), %zmm0
	vmovdqu64 %zmm0, (%rdi)
	addq	$64, %r12
	addq	$64, %rdi
	jmp	.Ld_grp

/* The same two as the chunk's last group: the block may be short, so
   they store exactly r13 bytes. */
	.align	16
.Ld_z1_last:
	movq	%rcx, %rbx
	movq	24(%rsp), %rdx
	leaq	(%rdi,%rdx), %rax
	cmpq	%r9, %rax
	ja	.Ld_espace
	movq	$-1, %rax
	bzhi	%rdx, %rax, %rax
	kmovq	%rax, %k1
	vmovdqu8 %zmm31, (%rdi){%k1}
	addq	%rdx, %rdi
	jmp	.Ld_chunk_next
	.align	16
.Ld_l1_last:
	movq	%rcx, %rbx
	movq	24(%rsp), %rdx
	leaq	(%rdi,%rdx), %rax
	cmpq	%r9, %rax
	ja	.Ld_espace
	leaq	(%r12,%rdx), %rax
	cmpq	%r8, %rax
	ja	.Ld_estream
	movq	$-1, %rax
	bzhi	%rdx, %rax, %rax
	kmovq	%rax, %k1
	vmovdqu8 (%r12), %zmm0{%k1}{z}
	vmovdqu8 %zmm0, (%rdi){%k1}
	addq	%rdx, %r12
	addq	%rdx, %rdi
	jmp	.Ld_chunk_next

	.align	16
.Ld_zrun:
	movzbl	%al, %ecx
	subl	$0xBF, %ecx
	shlq	$6, %rcx			/* zero bytes */
	leaq	1(%rbx), %rax
	cmpq	%rbp, %rax
	jae	.Ld_zrun_last
	movq	%rax, %rbx
.Ld_zrun_go:
	leaq	(%rdi,%rcx), %rax
	cmpq	%r9, %rax
	ja	.Ld_espace
	movq	%rcx, %rdx
	andq	$-64, %rdx
	xorl	%eax, %eax
	cmpq	$256, %rdx
	jb	.Ld_zrun_one
.Ld_zrun_four:
	vmovdqu64 %zmm31, (%rdi,%rax)
	vmovdqu64 %zmm31, 64(%rdi,%rax)
	vmovdqu64 %zmm31, 128(%rdi,%rax)
	vmovdqu64 %zmm31, 192(%rdi,%rax)
	addq	$256, %rax
	leaq	256(%rax), %r10
	cmpq	%rdx, %r10
	jbe	.Ld_zrun_four
.Ld_zrun_one:
	cmpq	%rdx, %rax
	jae	.Ld_zrun_tail
	vmovdqu64 %zmm31, (%rdi,%rax)
	addq	$64, %rax
	jmp	.Ld_zrun_one
.Ld_zrun_tail:
	movq	%rcx, %rax
	andl	$63, %eax
	jz	.Ld_zrun_done
	movq	$-1, %r10
	bzhi	%rax, %r10, %r10
	kmovq	%r10, %k1
	vmovdqu8 %zmm31, (%rdi,%rdx){%k1}
.Ld_zrun_done:
	addq	%rcx, %rdi
	cmpq	%rbp, %rbx
	jae	.Ld_chunk_next
	movq	(%rbx), %rax
	jmp	.Ld_grp
.Ld_zrun_last:
	movq	%rax, %rbx
	subq	$64, %rcx
	addq	24(%rsp), %rcx
	jmp	.Ld_zrun_go

/* Mask mode: the block's literal mask is in the stream. */
	.align	16
.Ld_mask:
	leaq	9(%rbx), %rcx
	cmpq	%rbp, %rcx
	ja	.Ld_estream
	movq	1(%rbx), %r11			/* keep */
	movl	$64, %r10d
	jb	.Ld_mask_fast
	movq	%rcx, %rbx
	movl	24(%rsp), %r10d			/* the chunk's last group */
	jmp	.Ld_emit
.Ld_mask_fast:
	movq	%rcx, %rbx
	cmpq	%r14, %rdi
	ja	.Ld_emit
	cmpq	32(%rsp), %r12
	ja	.Ld_emit
	popcntq	%r11, %rcx
	movq	(%rbx), %rax
	kmovq	%r11, %k1
	vpexpandb (%r12), %zmm0{%k1}{z}
	vmovdqu64 %zmm0, (%rdi)
	addq	%rcx, %r12
	addq	$64, %rdi
	jmp	.Ld_grp

/* A position list emitted exactly, when the sloppy paths have no room.
   The literal mask is the xor of ( ~0 << p ) over the list, complemented;
   with the list length known the terms can be laid out straight, three
   instructions each, which is the cheap way in and out of this path. */
	.align	16
.Ld_pos_exact5:					/* from the dispatch: rbx advanced */
	movl	$64, %edx
	jmp	.Ld_pos_wide
.Ld_pos_exact:					/* rcx = control after the group */
	movl	$64, %edx
	cmpq	%rbp, %rcx
	jb	.Ld_pos_adv
	movl	24(%rsp), %edx			/* the chunk's last group */
.Ld_pos_adv:
	movq	%rcx, %rbx
	cmpl	$2, %r10d
	je	.Ld_x3
	jb	.Ld_x12
	cmpl	$3, %r10d
	je	.Ld_x4
	jmp	.Ld_pos_wide
.Ld_x12:
	testl	%r10d, %r10d
	jnz	.Ld_x2

	.align	16
.Ld_x1:						/* one zero run, to the block end */
	movzbl	%al, %ecx
	movq	$-1, %r11
	shlx	%rcx, %r11, %rcx
	notq	%rcx
	movq	%rcx, %r11			/* keep */
	movq	%rdx, %r10			/* bs */
	jmp	.Ld_emit

	.align	16
.Ld_x2:
	movq	$-1, %r11
	movzbl	%al, %ecx
	shlx	%rcx, %r11, %rcx
	shrq	$8, %rax
	movzbl	%al, %r10d
	shlx	%r10, %r11, %r10
	xorq	%r10, %rcx
	notq	%rcx
	movq	%rcx, %r11
	movq	%rdx, %r10
	jmp	.Ld_emit

	.align	16
.Ld_x3:
	movq	$-1, %r11
	movzbl	%al, %ecx
	shlx	%rcx, %r11, %rcx
	shrq	$8, %rax
	movzbl	%al, %r10d
	shlx	%r10, %r11, %r10
	xorq	%r10, %rcx
	shrq	$8, %rax
	movzbl	%al, %r10d
	shlx	%r10, %r11, %r10
	xorq	%r10, %rcx
	notq	%rcx
	movq	%rcx, %r11
	movq	%rdx, %r10
	jmp	.Ld_emit

	.align	16
.Ld_x4:
	movq	$-1, %r11
	movzbl	%al, %ecx
	shlx	%rcx, %r11, %rcx
	shrq	$8, %rax
	movzbl	%al, %r10d
	shlx	%r10, %r11, %r10
	xorq	%r10, %rcx
	shrq	$8, %rax
	movzbl	%al, %r10d
	shlx	%r10, %r11, %r10
	xorq	%r10, %rcx
	shrq	$8, %rax
	movzbl	%al, %r10d
	shlx	%r10, %r11, %r10
	xorq	%r10, %rcx
	notq	%rcx
	movq	%rcx, %r11
	movq	%rdx, %r10
	jmp	.Ld_emit

/* Five or more positions: only a corrupt stream reaches this, since the
   coder switches to mask mode at five transitions.  bzhi leaves its
   source alone when the index is past 63, so a slot past the list
   contributes ~0 once its byte is forced to 0xFF, and with eight slots
   those cancel in pairs. */
	.align	16
.Ld_pos_wide:
	shll	$3, %r10d
	leal	8(%r10), %ecx			/* 8*t */
	movq	$-1, %r11
	bzhi	%rcx, %r11, %rcx
	notq	%rcx
	orq	%rcx, %rax			/* slots past the list: 0xFF */
	bzhi	%rax, %r11, %r10
	shrq	$8, %rax
	bzhi	%rax, %r11, %rcx
	xorq	%rcx, %r10
	shrq	$8, %rax
	bzhi	%rax, %r11, %rcx
	xorq	%rcx, %r10
	shrq	$8, %rax
	bzhi	%rax, %r11, %rcx
	xorq	%rcx, %r10
	shrq	$8, %rax
	bzhi	%rax, %r11, %rcx
	xorq	%rcx, %r10
	shrq	$8, %rax
	bzhi	%rax, %r11, %rcx
	xorq	%rcx, %r10
	shrq	$8, %rax
	bzhi	%rax, %r11, %rcx
	xorq	%rcx, %r10
	shrq	$8, %rax
	bzhi	%rax, %r11, %rcx
	xorq	%rcx, %r10
	notq	%r10
	movq	%r10, %r11			/* keep */
	movq	%rdx, %r10			/* bs */
	jmp	.Ld_emit

/* Emit one block: r11 keep mask, r10 block size.  A full block needs no
   mask on its store and no clip on its keep, which is the common case. */
.Ld_emit:
	cmpl	$64, %r10d
	jne	.Ld_emit_part
	leaq	64(%rdi), %rax
	cmpq	%r9, %rax
	ja	.Ld_espace
	popcntq	%r11, %rcx
	leaq	(%r12,%rcx), %rdx
	cmpq	%r8, %rdx
	ja	.Ld_estream
	kmovq	%r11, %k1
	vpexpandb (%r12), %zmm0{%k1}{z}
	vmovdqu64 %zmm0, (%rdi)
	movq	%rdx, %r12
	addq	$64, %rdi
	cmpq	%rbp, %rbx
	jae	.Ld_chunk_next
	movq	(%rbx), %rax
	jmp	.Ld_grp

	.align	16
.Ld_emit_part:					/* the blob's last block */
	leaq	(%rdi,%r10), %rax
	cmpq	%r9, %rax
	ja	.Ld_espace
	movq	$-1, %rax
	bzhi	%r10, %rax, %rax
	andq	%rax, %r11
	popcntq	%r11, %rcx
	leaq	(%r12,%rcx), %rdx
	cmpq	%r8, %rdx
	ja	.Ld_estream
	kmovq	%r11, %k1
	kmovq	%rax, %k2
	vpexpandb (%r12), %zmm0{%k1}{z}
	vmovdqu8 %zmm0, (%rdi){%k2}
	movq	%rdx, %r12
	addq	%r10, %rdi
	cmpq	%rbp, %rbx
	jae	.Ld_chunk_next
	movq	(%rbx), %rax
	jmp	.Ld_grp

.Ld_chunk_next:
	movq	%r12, %rdx
	cmpq	%r8, %rdx
	jb	.Ld_chunk
	movq	%rdi, %rax
	subq	(%rsp), %rax
.Ld_ret:
	vzeroupper
	addq	$ZLE14_DFRAME, %rsp
	.cfi_adjust_cfa_offset -ZLE14_DFRAME
	popq	%r14
	.cfi_adjust_cfa_offset -8
	popq	%r12
	.cfi_adjust_cfa_offset -8
	popq	%rbp
	.cfi_adjust_cfa_offset -8
	popq	%rbx
	.cfi_adjust_cfa_offset -8
	ret

/* The control bytes end within 8 of the blob: run the group loop over a
   padded copy instead. */
	.align	16
.Ld_ctlcopy:
	movq	%rbp, %rcx
	subq	%rbx, %rcx			/* ctl_sz */
	leaq	ZLE14_DCTL(%rsp), %r10
	xorl	%eax, %eax
.Ld_cc_loop:
	movq	%rcx, %r11
	subq	%rax, %r11
	cmpq	$64, %r11
	jb	.Ld_cc_tail
	vmovdqu64 (%rbx,%rax), %zmm0
	vmovdqu64 %zmm0, (%r10,%rax)
	addq	$64, %rax
	jmp	.Ld_cc_loop
.Ld_cc_tail:
	movq	$-1, %rdx
	bzhi	%r11, %rdx, %rdx
	kmovq	%rdx, %k1
	vmovdqu8 (%rbx,%rax), %zmm0{%k1}{z}
	vmovdqu64 %zmm0, (%r10,%rax)
	movq	%r10, %rbx
	leaq	(%r10,%rcx), %rbp
	movq	(%rbx), %rax
	jmp	.Ld_grp

.Ld_espace:
	movq	$-2, %rax
	jmp	.Ld_ret
.Ld_estream:
	movq	$-1, %rax
	jmp	.Ld_ret
.Ld_zero:
	xorl	%eax, %eax
	ret
	.cfi_endproc
	.size	fd_zle14_decompress,.-fd_zle14_decompress

#endif /* __x86_64__ && FD_HAS_AVX512 */
