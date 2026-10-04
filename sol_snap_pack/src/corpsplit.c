/* corpsplit - split a SNAPCORP1 corpus into per field streams so that a
   generic compressor can be measured on each of them separately.

   usage: corpsplit corpus outdir [--sort]

   --sort clusters the records by (owner, data_len, pubkey) first, which is
   the "we get to choose the account order" experiment.

   Streams written:
     slot.bin      zig zag varint delta of the slot
     pubkey.bin    raw 32 byte pubkeys
     owner.bin     varint index into owner_dict.bin (frequency ordered)
     owner_dict.bin
     lamports.bin  varint
     rent.bin      one byte class (0=u64 max, 1=zero, 2=other) + varint
     dlen.bin      varint
     exec.bin      one bit per record, packed
     data.bin      account data, concatenated
     whole.bin     the records as they came in (AoS), for the baseline */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

static u8 const * base;
static u64        fsz;
static inline u64 ld8( u8 const * p ){ u64 v; memcpy(&v,p,8); return v; }
static inline u64 mix( u64 h ){ h^=h>>33; h*=0xff51afd7ed558ccdUL; h^=h>>33; h*=0xc4ceb9fe1a85ec53UL; h^=h>>33; return h; }

typedef struct { u64 off; } idx_t;

typedef struct { u8 key[32]; u64 cnt; u64 id; int used; } own_t;
static own_t * ot; static u64 ot_cap, ot_mask, ot_cnt;
static void ot_init( u64 lg ){ ot_cap=1UL<<lg; ot_mask=ot_cap-1UL; ot=calloc(ot_cap,sizeof(own_t)); }
static own_t * ot_get( u8 const * k ){
  u64 i = mix(ld8(k))&ot_mask;
  for(;;){ own_t * e=ot+i;
    if(!e->used){ e->used=1; memcpy(e->key,k,32); ot_cnt++; return e; }
    if(!memcmp(e->key,k,32)) return e;
    i=(i+1UL)&ot_mask; }
}
static int cmp_own( void const * a, void const * b ){
  u64 x=(*(own_t * const *)a)->cnt, y=(*(own_t * const *)b)->cnt;
  return x<y?1:(x>y?-1:0);
}

typedef struct { FILE * f; u64 n; } strm_t;
static strm_t
sopen( char const * dir, char const * name ){
  char p[4096]; snprintf(p,sizeof(p),"%s/%s",dir,name);
  strm_t s; s.f = fopen(p,"wb"); s.n=0UL;
  if(!s.f){ perror(p); exit(1); }
  return s;
}
static void sput( strm_t * s, void const * p, size_t n ){ fwrite(p,1,n,s->f); s->n += n; }
static void svar( strm_t * s, u64 v ){
  u8 b[10]; int n=0;
  do { u8 x = v&0x7FU; v >>= 7; if(v) x |= 0x80U; b[n++]=x; } while(v);
  sput(s,b,(size_t)n);
}
static u64 zigzag( long long v ){ return ((u64)v<<1) ^ (u64)(v>>63); }

static u8 const * sort_base;
static int cmp_rec( void const * a, void const * b ){
  u64 oa = *(u64 const *)a, ob = *(u64 const *)b;
  u8 const * pa = sort_base+oa, * pb = sort_base+ob;
  int c = memcmp( pa+64, pb+64, 32 );              /* owner */
  if( c ) return c;
  u64 la = ld8(pa+24), lb = ld8(pb+24);            /* data_len */
  if( la!=lb ) return la<lb?-1:1;
  return memcmp( pa+32, pb+32, 32 );               /* pubkey */
}

int
main( int argc, char ** argv ){
  if( argc<3 ){ fprintf(stderr,"usage: %s corpus outdir [--sort]\n",argv[0]); return 1; }
  int do_sort = argc>3 && !strcmp(argv[3],"--sort");
  int fd = open(argv[1],O_RDONLY); if(fd<0){ perror("open"); return 1; }
  struct stat st; fstat(fd,&st); fsz=(u64)st.st_size;
  u8 * m = mmap(NULL,fsz,PROT_READ,MAP_PRIVATE,fd,0);
  if(m==MAP_FAILED){ perror("mmap"); return 1; }
  base=m; sort_base=m;
  if( memcmp(m,"SNAPCORP1",9) ){ fprintf(stderr,"bad magic\n"); return 1; }

  /* index the records */
  u64 cap = 1UL<<20, n = 0UL;
  u64 * idx = malloc(cap*8UL);
  ot_init( 20 );
  for( u64 off=16UL; off+104UL<=fsz; ){
    u64 dlen = ld8(base+off+24);
    if( off+104UL+dlen>fsz ) break;
    if( n==cap ){ cap*=2UL; idx=realloc(idx,cap*8UL); }
    idx[n++]=off;
    ot_get( base+off+64 )->cnt++;
    off += 104UL+dlen;
  }

  /* owner dictionary, most frequent first */
  {
    own_t ** v = malloc(ot_cnt*sizeof(own_t*)); u64 k=0UL;
    for( u64 i=0UL;i<ot_cap;i++ ) if(ot[i].used) v[k++]=ot+i;
    qsort(v,k,sizeof(own_t*),cmp_own);
    for( u64 i=0UL;i<k;i++ ) v[i]->id=i;
    strm_t d = sopen(argv[2],"owner_dict.bin");
    for( u64 i=0UL;i<k;i++ ) sput(&d,v[i]->key,32);
    fclose(d.f);
    fprintf(stderr,"records %lu, distinct owners %lu\n",n,k);
    free(v);
  }

  if( do_sort ) qsort( idx, n, 8UL, cmp_rec );

  strm_t s_slot=sopen(argv[2],"slot.bin"), s_pk=sopen(argv[2],"pubkey.bin"),
         s_own=sopen(argv[2],"owner.bin"), s_lam=sopen(argv[2],"lamports.bin"),
         s_rent=sopen(argv[2],"rent.bin"), s_len=sopen(argv[2],"dlen.bin"),
         s_exec=sopen(argv[2],"exec.bin"), s_data=sopen(argv[2],"data.bin"),
         s_whole=sopen(argv[2],"whole.bin");
  u64 prev_slot = 0UL;
  u8  ebits = 0U; int enb = 0;
  for( u64 i=0UL;i<n;i++ ){
    u8 const * p = base+idx[i];
    u64 slot=ld8(p), lam=ld8(p+8), rent=ld8(p+16), dlen=ld8(p+24);
    svar(&s_slot, zigzag((long long)(slot-prev_slot)) ); prev_slot=slot;
    sput(&s_pk,p+32,32);
    svar(&s_own, ot_get(p+64)->id );
    svar(&s_lam, lam);
    if( rent==(u64)-1 ) { u8 c=0; sput(&s_rent,&c,1); }
    else if( !rent )    { u8 c=1; sput(&s_rent,&c,1); }
    else { u8 c=2; sput(&s_rent,&c,1); svar(&s_rent,rent); }
    svar(&s_len,dlen);
    ebits |= (u8)((p[96]?1U:0U)<<enb);
    if( ++enb==8 ){ sput(&s_exec,&ebits,1); ebits=0U; enb=0; }
    if( dlen ) sput(&s_data,p+104,dlen);
    sput(&s_whole,p,104UL+dlen);
  }
  if( enb ) sput(&s_exec,&ebits,1);
  printf("stream,bytes\n");
  printf("slot,%lu\npubkey,%lu\nowner,%lu\nlamports,%lu\nrent,%lu\ndlen,%lu\nexec,%lu\ndata,%lu\nwhole,%lu\n",
         s_slot.n,s_pk.n,s_own.n,s_lam.n,s_rent.n,s_len.n,s_exec.n,s_data.n,s_whole.n);
  fclose(s_slot.f);fclose(s_pk.f);fclose(s_own.f);fclose(s_lam.f);fclose(s_rent.f);
  fclose(s_len.f);fclose(s_exec.f);fclose(s_data.f);fclose(s_whole.f);
  return 0;
}
