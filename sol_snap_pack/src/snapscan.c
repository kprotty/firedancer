/* snapscan - streaming statistics scanner for Solana snapshot tarballs.

   Reads an *uncompressed* snapshot tar stream on stdin, e.g.

     zstd -dc snapshot-*.tar.zst | snapscan [-s sample.corpus] [-m manifest.bin]

   and dumps statistics about the stored accounts relevant to designing a
   snapshot-specific compressor.  Nothing here depends on firedancer; the
   appendvec layout is the 136 byte header documented in
   src/discof/restore/utils/fd_ssparse.c:

     [  0,  8) write_version
     [  8, 16) data_len
     [ 16, 48) pubkey
     [ 48, 56) lamports
     [ 56, 64) rent_epoch
     [ 64, 96) owner
     [ 96, 97) executable
     [104,136) account hash
     [136,136+data_len) data, next entry starts at align_up(...,8) */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <sys/mman.h>
#include <immintrin.h>

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

/* ---------- input buffer over stdin ---------------------------------- */

static u8 *   ibuf;
static size_t ibuf_cap, ibuf_len, ibuf_pos;
static u64    total_in;

static void
ibuf_init( size_t cap ) {
  ibuf_cap = cap;
  ibuf     = malloc( cap );
  if( !ibuf ) { fprintf( stderr, "oom\n" ); exit(1); }
}

/* ensure at least need bytes live at ibuf+ibuf_pos.  returns bytes available
   (may be < need at EOF). */
static size_t
fill( size_t need ) {
  size_t avail = ibuf_len - ibuf_pos;
  if( avail>=need ) return avail;
  if( need>ibuf_cap ) {
    size_t ncap = ibuf_cap;
    while( ncap<need ) ncap *= 2;
    u8 * nb = realloc( ibuf, ncap );
    if( !nb ) { fprintf( stderr, "oom grow %zu\n", ncap ); exit(1); }
    ibuf = nb; ibuf_cap = ncap;
  }
  if( ibuf_pos ) { memmove( ibuf, ibuf+ibuf_pos, avail ); ibuf_len = avail; ibuf_pos = 0; }
  while( ibuf_len<need ) {
    ssize_t n = read( 0, ibuf+ibuf_len, ibuf_cap-ibuf_len );
    if( n<0 ) { if( errno==EINTR ) continue; perror("read"); exit(1); }
    if( n==0 ) break;
    ibuf_len += (size_t)n;
    total_in += (u64)n;
  }
  return ibuf_len - ibuf_pos;
}

/* skip n bytes of stream */
static void
skip( u64 n ) {
  while( n ) {
    size_t avail = ibuf_len - ibuf_pos;
    if( !avail ) { if( !fill( 1 ) ) return; avail = ibuf_len - ibuf_pos; }
    size_t take = avail < n ? avail : (size_t)n;
    ibuf_pos += take;
    n        -= take;
  }
}

/* ---------- pubkey hash table ---------------------------------------- */

typedef struct {
  u8  key[32];
  int used;
  u64 cnt;
  u64 data_bytes;
  u64 zero_bytes;
  u64 exec_cnt;
  u64 empty_cnt;     /* data_len==0 */
  u64 lamports;
  u64 dlen_min, dlen_max;
} pk_ent;

typedef struct {
  pk_ent * e;
  u64      cap, mask, cnt;
} pk_tbl;

static void
pk_init( pk_tbl * t, u64 lg ) {
  t->cap  = 1UL<<lg;
  t->mask = t->cap-1UL;
  t->cnt  = 0UL;
  t->e    = calloc( t->cap, sizeof(pk_ent) );
  if( !t->e ) { fprintf( stderr, "oom pk\n" ); exit(1); }
}

static u64
pk_load8( u8 const * p ) { u64 v; memcpy( &v, p, 8 ); return v; }

static inline u64 fd_ld8( u8 const * p ) { u64 v; memcpy( &v, p, 8 ); return v; }

/* returns slot index; pubkeys are already uniformly random */
static u64
pk_find( pk_tbl * t, u8 const * key ) {
  u64 h = pk_load8( key );
  u64 i = h & t->mask;
  for(;;) {
    pk_ent * e = t->e + i;
    if( !e->used ) {
      e->used = 1;
      memcpy( e->key, key, 32 );
      e->dlen_min = (u64)-1;
      t->cnt++;
      return i;
    }
    if( !memcmp( e->key, key, 32 ) ) return i;
    i = (i+1UL) & t->mask;
  }
}

/* ---------- (owner,data_len) group table ----------------------------- */

#define COL_OFF_MAX  256UL   /* per-offset byte histograms cover this prefix */
#define COL_GRP_MAX  192UL   /* how many groups get a column histogram      */
#define COL_MIN_CNT  2000UL  /* group must be this popular to earn one      */

typedef struct {
  u64  owner_slot;   /* index into owner table, +1 (0 == empty) */
  u64  dlen;
  u64  cnt;
  u64  zero_bytes;
  u64 (*col)[256];   /* COL_OFF_MAX x 256 or NULL */
} grp_ent;

static grp_ent * grp;
static u64       grp_cap, grp_mask, grp_cnt, grp_col_cnt;

static void
grp_init( u64 lg ) {
  grp_cap  = 1UL<<lg;
  grp_mask = grp_cap-1UL;
  grp      = calloc( grp_cap, sizeof(grp_ent) );
  if( !grp ) { fprintf( stderr, "oom grp\n" ); exit(1); }
}

static grp_ent *
grp_find( u64 owner_slot, u64 dlen ) {
  u64 h = (owner_slot*0x9E3779B97F4A7C15UL) ^ (dlen*0xC2B2AE3D27D4EB4FUL);
  h ^= h>>29; h *= 0xBF58476D1CE4E5B9UL; h ^= h>>32;
  u64 i = h & grp_mask;
  for(;;) {
    grp_ent * e = grp + i;
    if( !e->owner_slot ) {
      if( grp_cnt*4 > grp_cap*3 ) return NULL; /* table full-ish, stop tracking */
      e->owner_slot = owner_slot; e->dlen = dlen; grp_cnt++;
      return e;
    }
    if( e->owner_slot==owner_slot && e->dlen==dlen ) return e;
    i = (i+1UL) & grp_mask;
  }
}

/* ---------- exact value histogram (data_len, rent_epoch, ...) -------- */

typedef struct { u64 val, cnt; int used; } v_ent;
typedef struct { v_ent * e; u64 cap, mask, cnt; } v_tbl;

static void
v_init( v_tbl * t, u64 lg ) {
  t->cap = 1UL<<lg; t->mask = t->cap-1UL; t->cnt = 0UL;
  t->e = calloc( t->cap, sizeof(v_ent) );
  if( !t->e ) { fprintf( stderr, "oom v\n" ); exit(1); }
}

static void
v_add( v_tbl * t, u64 val, u64 n ) {
  u64 h = val*0x9E3779B97F4A7C15UL; h ^= h>>31;
  u64 i = h & t->mask;
  for(;;) {
    v_ent * e = t->e + i;
    if( !e->used ) {
      if( t->cnt*4 > t->cap*3 ) return;
      e->used = 1; e->val = val; e->cnt = n; t->cnt++; return;
    }
    if( e->val==val ) { e->cnt += n; return; }
    i = (i+1UL) & t->mask;
  }
}

/* ---------- optional: reference scan against the snapshot's own pubkeys ---- */

static u64 * rt;              /* fingerprints of every account pubkey */
static u64   rt_cap, rt_mask;
static u64   ref_probes, ref_hits, ref_hit_bytes, ref_zero_bytes, ref_left;

typedef struct { u64 fp, cnt; } rf_ent;
static rf_ent * rf;           /* frequency of each referenced key */
static u64      rf_cap, rf_mask, rf_cnt;

static inline u64 ref_fp( u8 const * p ) {
  u64 h = fd_ld8(p)*0x9E3779B97F4A7C15UL + fd_ld8(p+8)*0xC2B2AE3D27D4EB4FUL +
          fd_ld8(p+16)*0x165667B19E3779F9UL + fd_ld8(p+24)*0x27D4EB2F165667C5UL;
  h ^= h>>33; h *= 0xff51afd7ed558ccdUL; h ^= h>>33; h *= 0xc4ceb9fe1a85ec53UL; h ^= h>>33;
  return h ? h : 1UL;
}
static void ref_add( u64 fp ) {
  u64 i = fp & rt_mask;
  for(;;) { if( !rt[i] ) { rt[i] = fp; return; } if( rt[i]==fp ) return; i = (i+1UL)&rt_mask; }
}
static inline int ref_has( u64 fp ) {
  u64 i = fp & rt_mask;
  for(;;) { u64 v = rt[i]; if( !v ) return 0; if( v==fp ) return 1; i = (i+1UL)&rt_mask; }
}
/* ---------- optional: one pass unified dictionary, per segment (-G) -------
   Prices a codec that never pre-indexes the account set.  Every 32 byte key
   gets a rank the moment it first appears -- as an account's own pubkey, or
   as a reference to a key not seen yet, which costs 32 raw bytes once and is
   a rank thereafter.  The dictionary resets at each segment boundary, so a
   segment decodes independently: a zstd frame of blocks.  Several segment
   sizes run side by side in one pass over the snapshot. */


/* A preamble hot table: the K most referenced keys, shipped once with the
   snapshot (or inherited from the previous one, since the head of this
   distribution moves slowly).  A key in the table is resolvable in every
   segment without being re-emitted, which is exactly what segmentation
   otherwise destroys.  One shared table stores each key's frequency rank;
   a simulator with hot_k treats rank<hot_k as resident. */
static u64 * ht;                 /* fp at 2i, rank+1 at 2i+1 */
static u64   ht_cap, ht_mask, ht_n;

static void
hot_load( char const * path ) {
  int fd = open( path, O_RDONLY );
  if( fd<0 ) { perror("open hot"); exit(1); }
  off_t end = lseek( fd, 0, SEEK_END ); lseek( fd, 0, SEEK_SET );
  u64 n = (u64)end/16UL, lg = 20UL;
  while( (1UL<<lg) < n*3UL ) lg++;
  ht_cap = 1UL<<lg; ht_mask = ht_cap-1UL; ht_n = n;
  ht = calloc( ht_cap*2UL, 8UL );
  if( !ht ) { perror("calloc hot"); exit(1); }
  u64 (*buf)[2] = malloc( n*16UL );
  if( read( fd, buf, n*16UL ) != (ssize_t)(n*16UL) ) { perror("read hot"); exit(1); }
  close( fd );
  for( u64 r=0UL; r<n; r++ ) {          /* already sorted, most referenced first */
    u64 fp = buf[r][0], i = fp & ht_mask;
    for(;;) { if( !ht[2*i] ) { ht[2*i]=fp; ht[2*i+1]=r+1UL; break; }
              if( ht[2*i]==fp ) break; i=(i+1UL)&ht_mask; }
  }
  free( buf );
  fprintf( stderr, "hot table: %lu keys loaded\n", n );
}

/* 0 if absent, else rank+1 */
static inline u64
hot_rank( u64 fp ) {
  if( !ht ) return 0UL;
  u64 i = fp & ht_mask;
  for(;;) { u64 v=ht[2*i]; if( !v ) return 0UL; if( v==fp ) return ht[2*i+1]; i=(i+1UL)&ht_mask; }
}

static int cmp_freq_desc( void const * a, void const * b ){
  u64 const * x=a, * y=b;
  return x[1]<y[1] ? 1 : (x[1]>y[1] ? -1 : 0);
}

#define SEG_MAX 8
typedef struct {
  u64   seg_bytes;      /* input bytes per segment; 0 = never reset */
  u64   hot_k;          /* preamble table size; keys under it never escape */
  u64   hot_hit;
  u64 * t; u64 cap, mask, cnt;
  u64   fill;           /* input bytes in the current segment */
  u64   segs, peak;
  u64   hit, miss;      /* references resolved from the dict / escaped */
  u64   own_new, own_dup;  /* account pubkey: fresh, or already escaped here */
} seg_t;

static seg_t seg[SEG_MAX];
static int   seg_cnt = 0;

static void
seg_new( u64 seg_bytes, u64 hot_k, u64 total_hint ) {
  seg_t * s = seg + seg_cnt++;
  s->seg_bytes = seg_bytes; s->hot_k = hot_k;
  /* ~139 input bytes per inserted key (1.18e9 accounts + 1.87 refs each over
     475 GiB), capped by the number of distinct keys that exist at all. */
  u64 ins = seg_bytes ? seg_bytes/139UL : total_hint;
  if( ins > total_hint ) ins = total_hint;
  u64 lg = 20UL; while( (1UL<<lg) < ins*3UL ) lg++;
  if( lg > 33UL ) lg = 33UL;
  s->cap = 1UL<<lg; s->mask = s->cap-1UL;
  s->t = mmap( NULL, s->cap*8UL, PROT_READ|PROT_WRITE,
               MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0 );
  if( s->t==MAP_FAILED ) { perror("mmap seg"); exit(1); }
  madvise( s->t, s->cap*8UL, MADV_HUGEPAGE );
}

/* returns 1 if fp was already present */
static inline int
seg_put( seg_t * s, u64 fp ) {
  u64 i = fp & s->mask;
  for(;;) {
    u64 v = s->t[i];
    if( !v ) { s->t[i] = fp; s->cnt++; return 0; }
    if( v==fp ) return 1;
    i = (i+1UL) & s->mask;
  }
}

static void
seg_roll( seg_t * s ) {
  if( s->cnt > s->peak ) s->peak = s->cnt;
  s->segs++; s->cnt = 0UL; s->fill = 0UL;
  madvise( s->t, s->cap*8UL, MADV_DONTNEED );
}

/* called once per account, before its data is scanned */
static void
seg_acct( u8 const * pubkey, u64 rec_bytes ) {
  u64 fp = ref_fp( pubkey );
  for( int k=0; k<seg_cnt; k++ ) {
    seg_t * s = seg+k;
    if( s->seg_bytes && s->fill >= s->seg_bytes ) seg_roll( s );
    s->fill += rec_bytes;
    if( seg_put( s, fp ) ) s->own_dup++; else s->own_new++;
  }
}

/* called for every window that is a reference into the snapshot's key set */
static void
seg_ref( u64 fp ) {
  u64 r = hot_rank( fp );
  for( int k=0; k<seg_cnt; k++ ) {
    seg_t * s = seg+k;
    if( s->hot_k && r && r<=s->hot_k ) { s->hit++; s->hot_hit++; continue; }
    if( seg_put( s, fp ) ) s->hit++; else s->miss++;
  }
}

/* ---------- optional: one declared field, over the whole snapshot ---------
   The 1/128 sample cannot answer what a token account's owner field costs:
   each wallet's other token accounts are almost all missing from the sample,
   so the field looks like it has far more distinct values than it does, and
   every key that is not itself an account looks like a one off 32 byte
   escape.  This mode prices the field without sampling: how often it is a
   key the snapshot already carries, and -- for the rest -- how many distinct
   keys they are and what their frequency distribution is worth, which is
   exactly the value of a second dictionary holding the keys the codec has
   already escaped. */

static int   want_field = 0;             /* -W: field scan only, no window scan */
static u8    tok_owner[2][32];           /* SPL Token, Token-2022 */
static u64   fld_acct, fld_zero[2], fld_hit[2], fld_esc[2];
static rf_ent * ef;                      /* frequency of escaped owner keys */
static u64      ef_cap, ef_mask, ef_cnt, ef_dropped;

static void ef_bump( u64 fp ) {
  u64 i = fp & ef_mask;
  for(;;) {
    if( !ef[i].fp ) {
      if( ef_cnt*4UL>ef_cap*3UL ) { ef_dropped++; return; }
      ef[i].fp = fp; ef[i].cnt = 1UL; ef_cnt++; return;
    }
    if( ef[i].fp==fp ) { ef[i].cnt++; return; }
    i = (i+1UL)&ef_mask;
  }
}

static int
b58_dec32( char const * str, u8 * out ) {
  static char const * A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  u8 acc[32]; memset(acc,0,32);
  for( char const * p=str; *p; p++ ) {
    char const * q = strchr(A,*p); if( !q ) return -1;
    int carry = (int)(q-A);
    for( int i=31;i>=0;i-- ){ int t = acc[i]*58 + carry; acc[i]=(u8)t; carry=t>>8; }
    if( carry ) return -1;
  }
  memcpy(out,acc,32); return 0;
}

static void rf_bump( u64 fp ) {
  u64 i = fp & rf_mask;
  for(;;) {
    if( !rf[i].fp ) { if( rf_cnt*4UL>rf_cap*3UL ) return; rf[i].fp = fp; rf[i].cnt = 1UL; rf_cnt++; return; }
    if( rf[i].fp==fp ) { rf[i].cnt++; return; }
    i = (i+1UL)&rf_mask;
  }
}

/* ---------- global stats --------------------------------------------- */

static pk_tbl owners, pubkeys;
static int    track_pubkeys = 0;

static struct {
  u64 n_appendvec, n_acct, n_exec;
  u64 bytes_data, bytes_hdr, bytes_pad, bytes_other_files;
  u64 slot_min, slot_max;
  u64 lamports_zero, lamports_hist[65];
  u64 lamports_total;
  u64 rent_max, rent_zero, rent_other, rent_hist[65];
  u64 dlen_zero, dlen_hist[65];
  u64 wver_zero, wver_hist[65];
  u64 hash_zero, hash_nonzero;
  u64 data_zero_bytes, acct_all_zero, acct_no_trailing_zero;
  u64 trailing_zero_bytes;
  u64 zrun_bytes[6];    /* bytes covered by zero runs of len >=1,2,4,8,16,32 */
  u64 zrun_cnt[6];
  u64 byte_hist[256];
  u64 dup_slot_files;
} S;

static v_tbl dlen_tbl, rent_tbl, lam_tbl, slot_tbl;

static int
bitlen( u64 v ) { return v ? 64-__builtin_clzll(v) : 0; }

static int cmp_pk( void const * a, void const * b ) {
  u64 x = (*(pk_ent * const *)a)->cnt, y = (*(pk_ent * const *)b)->cnt;
  return x<y ? 1 : (x>y ? -1 : 0);
}
static int cmp_v( void const * a, void const * b ) {
  u64 x = (*(v_ent * const *)a)->cnt, y = (*(v_ent * const *)b)->cnt;
  return x<y ? 1 : (x>y ? -1 : 0);
}
static int cmp_grp( void const * a, void const * b ) {
  u64 x = (*(grp_ent * const *)a)->cnt, y = (*(grp_ent * const *)b)->cnt;
  return x<y ? 1 : (x>y ? -1 : 0);
}

/* ---------- corpus sampler ------------------------------------------- */

static int   key_fd    = -1;   /* dump of every account pubkey, 32B each */
static u8 *  key_buf;
static size_t key_len;
#define KEY_BUF_SZ (4UL<<20)

static int   av_fd     = -1;   /* whole-appendvec corpus (order preserving) */
static u64   av_every  = 256UL;
static u64   av_cap    = 8UL<<30;
static u64   av_written, av_cnt, av_vecs;
static int   av_take;

static int   samp_fd   = -1;
static u64   samp_mask = 63UL;    /* keep pubkey-hash & mask == 0 -> 1/64   */
static u64   samp_cap  = 8UL<<30; /* stop after this many bytes written     */
static u64   samp_written, samp_cnt;
static u8 *  samp_buf;
static size_t samp_len;
#define SAMP_BUF_SZ (4UL<<20)

static void
samp_flush( void ) {
  size_t off = 0;
  while( off<samp_len ) {
    ssize_t n = write( samp_fd, samp_buf+off, samp_len-off );
    if( n<0 ) { if( errno==EINTR ) continue; perror("write sample"); exit(1); }
    off += (size_t)n;
  }
  samp_written += samp_len;
  samp_len = 0;
}

static void
samp_put( void const * p, size_t n ) {
  if( n>=SAMP_BUF_SZ ) { samp_flush(); 
    size_t off=0; u8 const * b = p;
    while( off<n ) { ssize_t w = write( samp_fd, b+off, n-off );
      if( w<0 ) { if( errno==EINTR ) continue; perror("write sample"); exit(1); }
      off += (size_t)w; }
    samp_written += n; return; }
  if( samp_len+n > SAMP_BUF_SZ ) samp_flush();
  memcpy( samp_buf+samp_len, p, n ); samp_len += n;
}

static void
av_put( void const * p, size_t n ) {
  size_t off = 0UL; u8 const * b = p;
  while( off<n ) {
    ssize_t w = write( av_fd, b+off, n-off );
    if( w<0 ) { if( errno==EINTR ) continue; perror("write av"); exit(1); }
    off += (size_t)w;
  }
  av_written += n;
}

static void
av_acct( u64 slot, u8 const * hdr, u8 const * data, u64 dlen ) {
  if( av_fd<0 || !av_take || av_written>=av_cap ) return;
  u64 rec[4] = { slot, pk_load8(hdr+48), pk_load8(hdr+56), dlen };
  u8  tail[8]; memset( tail, 0, 8 ); tail[0] = hdr[96];
  av_put( rec, 32 ); av_put( hdr+16, 32 ); av_put( hdr+64, 32 ); av_put( tail, 8 );
  if( dlen ) av_put( data, dlen );
  av_cnt++;
}

/* record: slot,lamports,rent_epoch,data_len (u64 LE), pubkey[32], owner[32],
   executable, pad[7], data[data_len] */
static void
samp_acct( u64 slot, u8 const * hdr, u8 const * data, u64 dlen ) {
  if( samp_fd<0 || samp_written>=samp_cap ) return;
  u64 h = pk_load8( hdr+16 );
  h ^= h>>29; h *= 0xBF58476D1CE4E5B9UL; h ^= h>>32;
  if( h & samp_mask ) return;
  u64 rec[4] = { slot, pk_load8(hdr+48), pk_load8(hdr+56), dlen };
  u8  tail[8]; memset( tail, 0, 8 ); tail[0] = hdr[96];
  samp_put( rec, 32 );
  samp_put( hdr+16, 32 );
  samp_put( hdr+64, 32 );
  samp_put( tail, 8 );
  if( dlen ) samp_put( data, dlen );
  samp_cnt++;
}

static void
key_put( u8 const * pk ) {
  if( key_fd<0 ) return;
  memcpy( key_buf+key_len, pk, 32 ); key_len += 32UL;
  if( key_len==KEY_BUF_SZ ) {
    size_t off = 0UL;
    while( off<key_len ) {
      ssize_t w = write( key_fd, key_buf+off, key_len-off );
      if( w<0 ) { if( errno==EINTR ) continue; perror("write keys"); exit(1); }
      off += (size_t)w;
    }
    key_len = 0UL;
  }
}

/* ---------- account processing --------------------------------------- */

static u64 byte_hist4[4][256];
static u64 byte_hist_bytes;

static inline void
zrun_close( u64 run ) {
  int lg = bitlen(run)-1; if( lg>5 ) lg = 5;
  for( int k=0; k<=lg; k++ ) { S.zrun_bytes[k] += run; S.zrun_cnt[k]++; }
}


static void
do_acct( u64 slot, u8 const * hdr ) {
  u64 wver  = pk_load8( hdr+0  );
  u64 dlen  = pk_load8( hdr+8  );
  u64 lam   = pk_load8( hdr+48 );
  u64 rent  = pk_load8( hdr+56 );
  int exec  = hdr[96];
  u8 const * data = hdr+136;

  S.n_acct++;
  key_put( hdr+16UL );
  S.bytes_hdr  += 136UL;
  S.bytes_data += dlen;
  S.bytes_pad  += ((136UL+dlen+7UL)&~7UL) - (136UL+dlen);
  S.n_exec     += (u64)(exec!=0);

  S.lamports_total += lam;
  if( !lam ) S.lamports_zero++;
  S.lamports_hist[ bitlen(lam) ]++;
  v_add( &lam_tbl, lam, 1UL );

  if( rent==(u64)-1 )    S.rent_max++;
  else if( !rent )       S.rent_zero++;
  else                   S.rent_other++;
  S.rent_hist[ bitlen(rent) ]++;
  v_add( &rent_tbl, rent, 1UL );

  if( !dlen ) S.dlen_zero++;
  S.dlen_hist[ bitlen(dlen) ]++;
  v_add( &dlen_tbl, dlen, 1UL );

  if( !wver ) S.wver_zero++;
  S.wver_hist[ bitlen(wver) ]++;

  u64 hz = 0; for( int i=0; i<32; i++ ) hz |= hdr[104+i];
  if( hz ) S.hash_nonzero++; else S.hash_zero++;

  if( slot<S.slot_min ) S.slot_min = slot;
  if( slot>S.slot_max ) S.slot_max = slot;

  /* owner stats */
  u64 osl = pk_find( &owners, hdr+64 );
  pk_ent * oe = owners.e + osl;
  oe->cnt++;
  oe->data_bytes += dlen;
  oe->exec_cnt   += (u64)(exec!=0);
  oe->empty_cnt  += (u64)(dlen==0);
  oe->lamports   += lam;
  if( dlen < oe->dlen_min ) oe->dlen_min = dlen;
  if( dlen > oe->dlen_max ) oe->dlen_max = dlen;

  if( track_pubkeys ) { pk_ent * pe = pubkeys.e + pk_find( &pubkeys, hdr+16 ); pe->cnt++; }

  /* data content stats.  Zero runs are found 64 bytes at a time; the byte
     histogram and the columnar histograms are sampled, they are only used to
     estimate entropy. */
  u64 zbytes = 0UL;
  if( dlen ) {
    u64 run = 0UL;
    u64 i   = 0UL;
    for( ; i+64UL<=dlen; i+=64UL ) {
      __m512i v = _mm512_loadu_si512( (void const *)(data+i) );
      __mmask64 m = _mm512_cmpeq_epi8_mask( v, _mm512_setzero_si512() );
      if( m==~(__mmask64)0 ) { run += 64UL; zbytes += 64UL; continue; }
      if( !m ) { if( run ) { zrun_close( run ); run = 0UL; } continue; }
      zbytes += (u64)__builtin_popcountll( (u64)m );
      u64 mm = (u64)m, pos = 0UL;
      while( pos<64UL ) {
        if( (mm>>pos)&1UL ) {
          u64 t   = ~(mm>>pos);
          u64 len = t ? (u64)__builtin_ctzll( t ) : 64UL-pos;
          run += len; pos += len;
          if( pos<64UL ) { zrun_close( run ); run = 0UL; }
        } else {
          if( run ) { zrun_close( run ); run = 0UL; }
          u64 t   = mm>>pos;
          u64 len = t ? (u64)__builtin_ctzll( t ) : 64UL-pos;
          pos += len;
        }
      }
    }
    for( ; i<dlen; i++ ) {
      if( !data[i] ) { run++; zbytes++; }
      else if( run ) { zrun_close( run ); run = 0UL; }
    }
    /* the still-open run at the end is exactly the trailing zero run */
    S.trailing_zero_bytes += run;
    if( !run ) S.acct_no_trailing_zero++;
    if( run==dlen ) S.acct_all_zero++;
    if( run ) zrun_close( run );
    S.data_zero_bytes += zbytes;

    if( !(S.n_acct & 31UL) ) {
      u64 n = dlen<4096UL ? dlen : 4096UL;
      for( u64 j=0UL; j<n; j++ ) byte_hist4[ j&3UL ][ data[j] ]++;
      byte_hist_bytes += n;
    }
  }

  oe->zero_bytes += zbytes;

  /* An SPL token *account* only: 165 bytes for SPL Token, and for Token-2022
     either exactly 165 bytes or an account_type byte of 2 at offset 165.  A
     mint is 82 bytes and holds a COption tag at offset 0, so including one
     would put 32 bytes of supply and flags into the owner field's statistics. */
  if( seg_cnt ) seg_acct( hdr+16UL, 136UL+dlen );

  if( want_field && rt &&
      ( ( !memcmp( hdr+64, tok_owner[0], 32 ) && dlen==165UL ) ||
        ( !memcmp( hdr+64, tok_owner[1], 32 ) &&
          ( dlen==165UL || ( dlen>165UL && data[165]==2 ) ) ) ) ) {
    /* offset 0 is the mint, offset 32 the owner (wallet).  Only the owner's
       escapes go into the frequency table: the mint is resident ~100% of the
       time and would only dilute it. */
    u8 z32[32]; memset( z32, 0, 32 );
    fld_acct++;
    for( int f=0; f<2; f++ ) {
      u8 const * v = data + 32*f;
      if( !memcmp( v, z32, 32 ) ) { fld_zero[f]++; continue; }
      u64 fp = ref_fp( v );
      if( ref_has( fp ) ) { fld_hit[f]++; continue; }
      fld_esc[f]++;
      if( f==1 ) { ef_bump( fp ); if( seg_cnt ) seg_ref( fp ); }
    }
  }

  if( rt && !want_field && dlen>=32UL ) {
    u8 z32[32]; memset( z32, 0, 32 );
    u64 i = 0UL;
    while( i+32UL<=dlen ) {
      if( !fd_ld8(data+i) && !memcmp( data+i, z32, 32 ) ) { ref_zero_bytes += 32UL; i += 32UL; continue; }
      ref_probes++;
      u64 fp = ref_fp( data+i );
      if( ref_has( fp ) ) { ref_hits++; ref_hit_bytes += 32UL; rf_bump( fp );
                            if( seg_cnt ) seg_ref( fp );
                            i += 32UL; }
      else { ref_left += 4UL; i += 4UL; }
    }
    ref_left += dlen-i;
  }

  /* (owner,len) group + column histograms */
  grp_ent * g = grp_find( osl+1UL, dlen );
  if( g ) {
    g->cnt++;
    g->zero_bytes += zbytes;
    if( !g->col && g->cnt>=COL_MIN_CNT && grp_col_cnt<COL_GRP_MAX && dlen ) {
      g->col = calloc( COL_OFF_MAX, sizeof(u64[256]) );
      if( g->col ) grp_col_cnt++;
    }
    if( g->col && !(g->cnt & 3UL) ) {
      u64 n = dlen<COL_OFF_MAX ? dlen : COL_OFF_MAX;
      for( u64 i=0UL; i<n; i++ ) g->col[i][ data[i] ]++;
    }
  }
}

/* ---------- appendvec / tar ------------------------------------------ */

static u64
oct( char const * p, size_t n ) {
  u64 v = 0UL;
  for( size_t i=0UL; i<n; i++ ) {
    char c = p[i];
    if( c==' ' || c=='\0' ) { if( v ) break; else continue; }
    if( c<'0' || c>'7' ) break;
    v = v*8UL + (u64)(c-'0');
  }
  return v;
}

static void
appendvec( u64 slot, u64 fsz ) {
  u64 off = 0UL;
  S.n_appendvec++;
  v_add( &slot_tbl, slot, 1UL );
  av_take = ( av_fd>=0 ) && ( (S.n_appendvec % av_every)==0UL );
  if( av_take ) av_vecs++;
  while( off+136UL<=fsz ) {
    if( fill( 136UL )<136UL ) { fprintf( stderr, "truncated appendvec header\n" ); exit(1); }
    u64 dlen = pk_load8( ibuf+ibuf_pos+8UL );
    if( dlen>(16UL<<20) || off+136UL+dlen>fsz ) break;    /* tail slack */
    u64 need = 136UL+dlen;
    if( fill( need )<need ) { fprintf( stderr, "truncated appendvec data\n" ); exit(1); }
    u8 const * hdr = ibuf+ibuf_pos;
    do_acct( slot, hdr );
    samp_acct( slot, hdr, hdr+136UL, dlen );
    av_acct( slot, hdr, hdr+136UL, dlen );
    u64 adv = (need+7UL)&~7UL;
    if( off+adv>fsz ) adv = fsz-off;
    if( fill( adv )<adv ) { fprintf( stderr, "truncated appendvec pad\n" ); exit(1); }
    ibuf_pos += adv;
    off      += adv;
  }
  skip( fsz-off );
}

static int   want_manifest = 0;
static char  manifest_path[4096];

static void
dump_file( u64 fsz, char const * path ) {
  int fd = open( path, O_CREAT|O_TRUNC|O_WRONLY, 0644 );
  if( fd<0 ) { perror("open manifest"); exit(1); }
  u64 left = fsz;
  while( left ) {
    size_t avail = fill( 1UL );
    if( !avail ) break;
    size_t take = avail < left ? avail : (size_t)left;
    size_t off  = 0UL;
    while( off<take ) {
      ssize_t n = write( fd, ibuf+ibuf_pos+off, take-off );
      if( n<0 ) { if( errno==EINTR ) continue; perror("write manifest"); exit(1); }
      off += (size_t)n;
    }
    ibuf_pos += take; left -= take;
  }
  close( fd );
}

int
main( int argc, char ** argv ) {
  char const * samp_path = NULL;
  char const * av_path = NULL;
  char const * key_path = NULL;
  char const * ref_path = NULL;
  char const * seg_spec = NULL;
  char const * hot_path = NULL;
  char const * freq_path = NULL;
  u64 acct_limit = 0UL;
  for( int i=1; i<argc; i++ ) {
    if( !strcmp(argv[i],"-s") && i+1<argc ) samp_path = argv[++i];
    else if( !strcmp(argv[i],"-r") && i+1<argc ) samp_mask = (1UL<<strtoul(argv[++i],NULL,0))-1UL;
    else if( !strcmp(argv[i],"-c") && i+1<argc ) samp_cap = strtoull(argv[++i],NULL,0);
    else if( !strcmp(argv[i],"-m") && i+1<argc ) { want_manifest=1; snprintf(manifest_path,sizeof(manifest_path),"%s",argv[++i]); }
    else if( !strcmp(argv[i],"-n") && i+1<argc ) acct_limit = strtoull(argv[++i],NULL,0);
    else if( !strcmp(argv[i],"-A") && i+1<argc ) av_path = argv[++i];
    else if( !strcmp(argv[i],"-a") && i+1<argc ) av_every = strtoull(argv[++i],NULL,0);
    else if( !strcmp(argv[i],"-C") && i+1<argc ) av_cap = strtoull(argv[++i],NULL,0);
    else if( !strcmp(argv[i],"-k") && i+1<argc ) key_path = argv[++i];
    else if( !strcmp(argv[i],"-R") && i+1<argc ) ref_path = argv[++i];
    else if( !strcmp(argv[i],"-W") && i+1<argc ) { ref_path = argv[++i]; want_field = 1; }
    else if( !strcmp(argv[i],"-p") ) track_pubkeys = 1;
    else if( !strcmp(argv[i],"-G") && i+1<argc ) seg_spec = argv[++i];
    else if( !strcmp(argv[i],"-P") && i+1<argc ) hot_path = argv[++i];
    else if( !strcmp(argv[i],"-F") && i+1<argc ) freq_path = argv[++i];
    else { fprintf( stderr,
      "usage: %s [options] < uncompressed-snapshot.tar\n"
      "  -s FILE  write a pubkey-hash sampled corpus\n"
      "  -r N     sample rate for -s is 1/2^N (default 6)\n"
      "  -c N     cap the -s corpus at N bytes\n"
      "  -A FILE  write a corpus of whole appendvecs (preserves on-disk order)\n"
      "  -a N     take every Nth appendvec for -A (default 256)\n"
      "  -C N     cap the -A corpus at N bytes\n"
      "  -k FILE  dump every account pubkey, 32 bytes each\n"
      "  -R FILE  reference scan: match account data against the keys in FILE\n"
      "  -W FILE  like -R, but price only the SPL token mint/owner fields, and\n"
      "           count the distinct owner keys that are not accounts\n"
      "  -m FILE  dump the manifest\n"
      "  -n N     stop after N accounts\n"
      "  -p       count distinct pubkeys (needs ~110 GiB of RAM)\n"
      "  -G LIST  with -R: simulate a one pass unified dictionary reset every\n"
      "           N MiB of input, for each N[:K] in LIST (0 = never reset; K =\n"
      "           preamble hot table size, needs -P)\n"
      "  -P FILE  hot table: (fingerprint,count) pairs, most referenced first\n"
      "  -F FILE  dump the exact global (fingerprint,count) reference table\n", argv[0] ); return 1; }
  }

  if( ref_path ) {
    int rfd = open( ref_path, O_RDONLY );
    if( rfd<0 ) { perror("open refkeys"); return 1; }
    off_t end = lseek( rfd, 0, SEEK_END ); lseek( rfd, 0, SEEK_SET );
    u64 nkeys = (u64)end/32UL, lg = 20UL;
    while( (1UL<<lg) < nkeys*2UL ) lg++;
    rt_cap = 1UL<<lg; rt_mask = rt_cap-1UL;
    rt = mmap( NULL, rt_cap*8UL, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0 );
    if( rt==MAP_FAILED ) { perror("mmap rt"); return 1; }
    madvise( rt, rt_cap*8UL, MADV_HUGEPAGE );
    if( seg_spec ) {
      char buf[256]; snprintf(buf,sizeof(buf),"%s",seg_spec);
      if( hot_path ) hot_load( hot_path );
      for( char * q=strtok(buf,","); q && seg_cnt<SEG_MAX; q=strtok(NULL,",") ) {
        char * c = strchr( q, ':' );
        u64 kk = c ? strtoull(c+1,NULL,0) : 0UL;
        if( c ) *c = 0;
        seg_new( strtoull(q,NULL,0)*(1UL<<20), kk, 1500000000UL );
      }
      fprintf( stderr, "seg: %d simulators\n", seg_cnt );
    }
    rf_cap = 1UL<<29; rf_mask = rf_cap-1UL;
    rf = mmap( NULL, rf_cap*sizeof(rf_ent), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0 );
    if( rf==MAP_FAILED ) { perror("mmap rf"); return 1; }
    madvise( rf, rf_cap*sizeof(rf_ent), MADV_HUGEPAGE );
    fprintf( stderr, "loading %lu reference keys into 2^%lu slots\n", nkeys, lg );
    u8 * kb = malloc( 32UL<<20 );
    for(;;) {
      ssize_t got = read( rfd, kb, 32UL<<20 );
      if( got<0 ) { perror("read refkeys"); return 1; }
      if( !got ) break;
      for( ssize_t j=0; j+32<=got; j+=32 ) ref_add( ref_fp( kb+j ) );
    }
    free( kb ); close( rfd );
    fprintf( stderr, "reference keys loaded\n" );
    if( want_field ) {
      if( b58_dec32( "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", tok_owner[0] ) ||
          b58_dec32( "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb", tok_owner[1] ) ) {
        fprintf( stderr, "bad token owner constant\n" ); return 1;
      }
      ef_cap = 1UL<<31; ef_mask = ef_cap-1UL;
      ef = mmap( NULL, ef_cap*sizeof(rf_ent), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0 );
      if( ef==MAP_FAILED ) { perror("mmap ef"); return 1; }
      madvise( ef, ef_cap*sizeof(rf_ent), MADV_HUGEPAGE );
    }
  }

  ibuf_init( 64UL<<20 );
  pk_init( &owners, 20 );
  if( track_pubkeys ) pk_init( &pubkeys, 30 );
  grp_init( 22 );
  v_init( &dlen_tbl, 20 );
  v_init( &rent_tbl, 20 );
  v_init( &lam_tbl,  22 );
  v_init( &slot_tbl, 22 );
  S.slot_min = (u64)-1;

  if( key_path ) {
    key_fd = open( key_path, O_CREAT|O_TRUNC|O_WRONLY, 0644 );
    if( key_fd<0 ) { perror("open keys"); return 1; }
    key_buf = malloc( KEY_BUF_SZ );
  }

  if( av_path ) {
    av_fd = open( av_path, O_CREAT|O_TRUNC|O_WRONLY, 0644 );
    if( av_fd<0 ) { perror("open av"); return 1; }
    char magic[16]; memset(magic,0,16); memcpy(magic,"SNAPCORP1",9);
    av_put( magic, 16 );
  }

  if( samp_path ) {
    samp_fd = open( samp_path, O_CREAT|O_TRUNC|O_WRONLY, 0644 );
    if( samp_fd<0 ) { perror("open sample"); return 1; }
    samp_buf = malloc( SAMP_BUF_SZ );
    char magic[16]; memset(magic,0,16); memcpy(magic,"SNAPCORP1",9);
    samp_put( magic, 16 );
  }

  u64 zero_hdrs = 0UL;
  for(;;) {
    if( fill( 512UL )<512UL ) break;
    u8 const * th = ibuf+ibuf_pos;
    int all_zero = 1;
    for( int i=0; i<512; i++ ) if( th[i] ) { all_zero = 0; break; }
    if( all_zero ) { ibuf_pos += 512UL; if( ++zero_hdrs>=2UL ) break; continue; }
    zero_hdrs = 0UL;

    char name[101]; memcpy( name, th, 100 ); name[100] = '\0';
    u64  fsz  = oct( (char const *)th+124, 12 );
    char type = (char)th[156];
    ibuf_pos += 512UL;

    if( type=='5' || type=='L' || type=='K' ) { skip( (fsz+511UL)&~511UL ); continue; }

    if( !strncmp( name, "accounts/", 9 ) ) {
      u64 slot = strtoull( name+9, NULL, 10 );
      appendvec( slot, fsz );
      skip( ((fsz+511UL)&~511UL) - fsz );
    } else {
      S.bytes_other_files += fsz;
      if( want_manifest && !strncmp( name, "snapshots/", 10 ) && strncmp( name, "snapshots/status_cache", 22 ) ) {
        char p[4200]; snprintf( p, sizeof(p), "%s", manifest_path );
        dump_file( fsz, p );
        skip( ((fsz+511UL)&~511UL) - fsz );
      } else {
        skip( (fsz+511UL)&~511UL );
      }
    }
    if( acct_limit && S.n_acct>=acct_limit ) break;
    if( (S.n_appendvec & 0xFFFUL)==0UL ) {
      fprintf( stderr, "\r%.1f GiB in, %lu appendvecs, %lu accounts, %.1f GiB data",
               (double)total_in/1073741824.0, S.n_appendvec, S.n_acct,
               (double)S.bytes_data/1073741824.0 );
      fflush( stderr );
    }
  }
  fprintf( stderr, "\n" );
  if( samp_fd>=0 ) { samp_flush(); close( samp_fd ); }
  if( av_fd>=0 ) close( av_fd );
  if( key_fd>=0 ) {
    size_t off = 0UL;
    while( off<key_len ) {
      ssize_t w = write( key_fd, key_buf+off, key_len-off );
      if( w<0 ) { if( errno==EINTR ) continue; perror("write keys"); exit(1); }
      off += (size_t)w;
    }
    close( key_fd );
  }

  for( int b=0; b<256; b++ ) S.byte_hist[b] = byte_hist4[0][b]+byte_hist4[1][b]+byte_hist4[2][b]+byte_hist4[3][b];

  /* ---------- report ---------- */
  printf( "{\n" );
  printf( "  \"stream_bytes\": %lu,\n", total_in );
  printf( "  \"appendvecs\": %lu,\n", S.n_appendvec );
  printf( "  \"accounts\": %lu,\n", S.n_acct );
  printf( "  \"bytes_hdr\": %lu,\n", S.bytes_hdr );
  printf( "  \"bytes_data\": %lu,\n", S.bytes_data );
  printf( "  \"bytes_pad\": %lu,\n", S.bytes_pad );
  printf( "  \"bytes_other_files\": %lu,\n", S.bytes_other_files );
  printf( "  \"slot_min\": %lu,\n", S.slot_min );
  printf( "  \"slot_max\": %lu,\n", S.slot_max );
  printf( "  \"distinct_slots\": %lu,\n", slot_tbl.cnt );
  printf( "  \"distinct_owners\": %lu,\n", owners.cnt );
  if( track_pubkeys ) printf( "  \"distinct_pubkeys\": %lu,\n", pubkeys.cnt );
  printf( "  \"executable\": %lu,\n", S.n_exec );
  printf( "  \"lamports_zero\": %lu,\n", S.lamports_zero );
  printf( "  \"lamports_total\": %lu,\n", S.lamports_total );
  printf( "  \"rent_epoch_max\": %lu,\n", S.rent_max );
  printf( "  \"rent_epoch_zero\": %lu,\n", S.rent_zero );
  printf( "  \"rent_epoch_other\": %lu,\n", S.rent_other );
  printf( "  \"data_len_zero\": %lu,\n", S.dlen_zero );
  printf( "  \"write_version_zero\": %lu,\n", S.wver_zero );
  printf( "  \"hash_zero\": %lu,\n", S.hash_zero );
  printf( "  \"hash_nonzero\": %lu,\n", S.hash_nonzero );
  printf( "  \"data_zero_bytes\": %lu,\n", S.data_zero_bytes );
  printf( "  \"trailing_zero_bytes\": %lu,\n", S.trailing_zero_bytes );
  printf( "  \"acct_all_zero_data\": %lu,\n", S.acct_all_zero );
  printf( "  \"acct_no_trailing_zero\": %lu,\n", S.acct_no_trailing_zero );

  printf( "  \"zero_runs\": [" );
  for( int k=0; k<6; k++ ) printf( "%s{\"min_len\":%d,\"runs\":%lu,\"bytes\":%lu}", k?",":"", 1<<k, S.zrun_cnt[k], S.zrun_bytes[k] );
  printf( "],\n" );

  printf( "  \"lamports_bitlen\": [" );
  for( int i=0; i<65; i++ ) printf( "%s%lu", i?",":"", S.lamports_hist[i] );
  printf( "],\n" );
  printf( "  \"rent_epoch_bitlen\": [" );
  for( int i=0; i<65; i++ ) printf( "%s%lu", i?",":"", S.rent_hist[i] );
  printf( "],\n" );
  printf( "  \"data_len_bitlen\": [" );
  for( int i=0; i<65; i++ ) printf( "%s%lu", i?",":"", S.dlen_hist[i] );
  printf( "],\n" );
  printf( "  \"write_version_bitlen\": [" );
  for( int i=0; i<65; i++ ) printf( "%s%lu", i?",":"", S.wver_hist[i] );
  printf( "],\n" );
  printf( "  \"byte_hist_bytes\": %lu,\n", byte_hist_bytes );
  printf( "  \"byte_hist\": [" );
  for( int i=0; i<256; i++ ) printf( "%s%lu", i?",":"", S.byte_hist[i] );
  printf( "],\n" );

  /* top owners */
  {
    u64 n = owners.cnt, k = 0UL;
    pk_ent ** v = malloc( n*sizeof(pk_ent*) );
    for( u64 i=0UL; i<owners.cap; i++ ) if( owners.e[i].used ) v[k++] = owners.e+i;
    qsort( v, k, sizeof(pk_ent*), cmp_pk );
    u64 top = k<64UL ? k : 64UL;
    printf( "  \"top_owners\": [\n" );
    for( u64 i=0UL; i<top; i++ ) {
      printf( "    {\"owner\":\"" );
      for( int j=0; j<32; j++ ) printf( "%02x", v[i]->key[j] );
      printf( "\",\"cnt\":%lu,\"data_bytes\":%lu,\"zero_bytes\":%lu,\"empty\":%lu,\"exec\":%lu,\"dlen_min\":%lu,\"dlen_max\":%lu,\"lamports\":%lu}%s\n",
              v[i]->cnt, v[i]->data_bytes, v[i]->zero_bytes, v[i]->empty_cnt, v[i]->exec_cnt,
              v[i]->dlen_min==(u64)-1?0UL:v[i]->dlen_min, v[i]->dlen_max, v[i]->lamports, i+1UL<top?",":"" );
    }
    printf( "  ],\n" );
    free( v );
  }

  /* top data lens */
  {
    u64 k = 0UL;
    v_ent ** v = malloc( dlen_tbl.cnt*sizeof(v_ent*) );
    for( u64 i=0UL; i<dlen_tbl.cap; i++ ) if( dlen_tbl.e[i].used ) v[k++] = dlen_tbl.e+i;
    qsort( v, k, sizeof(v_ent*), cmp_v );
    u64 top = k<48UL ? k : 48UL;
    printf( "  \"distinct_data_len\": %lu,\n", dlen_tbl.cnt );
    printf( "  \"top_data_len\": [" );
    for( u64 i=0UL; i<top; i++ ) printf( "%s{\"len\":%lu,\"cnt\":%lu}", i?",":"", v[i]->val, v[i]->cnt );
    printf( "],\n" );
    free( v );
  }

  /* top rent epochs / lamports */
  {
    u64 k = 0UL;
    v_ent ** v = malloc( rent_tbl.cnt*sizeof(v_ent*) );
    for( u64 i=0UL; i<rent_tbl.cap; i++ ) if( rent_tbl.e[i].used ) v[k++] = rent_tbl.e+i;
    qsort( v, k, sizeof(v_ent*), cmp_v );
    u64 top = k<24UL ? k : 24UL;
    printf( "  \"distinct_rent_epoch\": %lu,\n", rent_tbl.cnt );
    printf( "  \"top_rent_epoch\": [" );
    for( u64 i=0UL; i<top; i++ ) printf( "%s{\"v\":%lu,\"cnt\":%lu}", i?",":"", v[i]->val, v[i]->cnt );
    printf( "],\n" );
    free( v );

    k = 0UL;
    v = malloc( lam_tbl.cnt*sizeof(v_ent*) );
    for( u64 i=0UL; i<lam_tbl.cap; i++ ) if( lam_tbl.e[i].used ) v[k++] = lam_tbl.e+i;
    qsort( v, k, sizeof(v_ent*), cmp_v );
    top = k<24UL ? k : 24UL;
    printf( "  \"distinct_lamports\": %lu,\n", lam_tbl.cnt );
    printf( "  \"top_lamports\": [" );
    for( u64 i=0UL; i<top; i++ ) printf( "%s{\"v\":%lu,\"cnt\":%lu}", i?",":"", v[i]->val, v[i]->cnt );
    printf( "],\n" );
    free( v );
  }

  /* top (owner,len) groups with column entropy */
  {
    u64 k = 0UL;
    grp_ent ** v = malloc( grp_cnt*sizeof(grp_ent*) );
    for( u64 i=0UL; i<grp_cap; i++ ) if( grp[i].owner_slot ) v[k++] = grp+i;
    qsort( v, k, sizeof(grp_ent*), cmp_grp );
    u64 top = k<64UL ? k : 64UL;
    printf( "  \"distinct_owner_len_groups\": %lu,\n", grp_cnt );
    printf( "  \"top_groups\": [\n" );
    for( u64 i=0UL; i<top; i++ ) {
      grp_ent * g = v[i];
      pk_ent * oe = owners.e + (g->owner_slot-1UL);
      printf( "    {\"owner\":\"" );
      for( int j=0; j<32; j++ ) printf( "%02x", oe->key[j] );
      printf( "\",\"len\":%lu,\"cnt\":%lu,\"zero_frac\":%.4f", g->dlen, g->cnt,
              g->dlen&&g->cnt ? (double)g->zero_bytes/((double)g->dlen*(double)g->cnt) : 0.0 );
      if( g->col ) {
        /* per-offset (columnar) entropy over the first COL_OFF_MAX bytes */
        u64 n = g->dlen<COL_OFF_MAX ? g->dlen : COL_OFF_MAX;
        double bits = 0.0;
        double per_off[COL_OFF_MAX];
        for( u64 o=0UL; o<n; o++ ) {
          double tot = 0.0, h = 0.0;
          for( int b=0; b<256; b++ ) tot += (double)g->col[o][b];
          if( tot>0.0 ) for( int b=0; b<256; b++ ) {
            double p = (double)g->col[o][b]/tot;
            if( p>0.0 ) h -= p*log2(p);
          }
          per_off[o] = h; bits += h;
        }
        printf( ",\"col_bits_per_rec\":%.2f,\"col_bytes_per_rec\":%.2f,\"col_prefix\":%lu",
                bits, bits/8.0, n );
        printf( ",\"col_entropy\":[" );
        for( u64 o=0UL; o<n; o++ ) printf( "%s%.2f", o?",":"", per_off[o] );
        printf( "]" );
      }
      printf( "}%s\n", i+1UL<top?",":"" );
    }
    printf( "  ],\n" );
    free( v );
  }

  if( freq_path && rf ) {
    u64 m=0UL; for( u64 i=0UL;i<rf_cap;i++ ) if( rf[i].fp ) m++;
    u64 (*hc)[2] = malloc(m*16UL); u64 k=0UL;
    for( u64 i=0UL;i<rf_cap;i++ ) if( rf[i].fp ) { hc[k][0]=rf[i].fp; hc[k][1]=rf[i].cnt; k++; }
    qsort( hc, m, 16UL, cmp_freq_desc );
    FILE * ff = fopen( freq_path, "wb" );
    if( ff ) { fwrite( hc, 16UL, m, ff ); fclose( ff );
               fprintf( stderr, "freq table: %lu keys -> %s\n", m, freq_path ); }
    free( hc );
  }

  if( seg_cnt ) {
    printf( "  \"segsim\": [\n" );
    for( int k=0; k<seg_cnt; k++ ) {
      seg_t * s = seg+k;
      if( s->cnt > s->peak ) s->peak = s->cnt;
      u64 refs = s->hit + s->miss;
      printf( "    {\"seg_mib\":%lu,\"segments\":%lu,\"peak_dict\":%lu,"
              "\"ref_hit\":%lu,\"ref_miss\":%lu,\"hit_rate\":%.5f,"
              "\"own_new\":%lu,\"own_dup\":%lu,\"hot_k\":%lu,\"hot_hit\":%lu}%s\n",
              s->seg_bytes>>20, s->segs+1UL, s->peak, s->hit, s->miss,
              refs ? (double)s->hit/(double)refs : 0.0,
              s->own_new, s->own_dup, s->hot_k, s->hot_hit, k+1<seg_cnt?",":"" );
    }
    printf( "  ],\n" );
  }

  if( rt ) {
    double H = 0.0; u64 distinct = 0UL;
    for( u64 i=0UL;i<rf_cap;i++ ) if( rf[i].fp ) {
      distinct++;
      double p = (double)rf[i].cnt/(double)ref_hits;
      H -= p*log2(p);
    }
    printf( "  \"ref_probes\": %lu,\n", ref_probes );
    printf( "  \"ref_hits\": %lu,\n", ref_hits );
    printf( "  \"ref_hit_bytes\": %lu,\n", ref_hit_bytes );
    printf( "  \"ref_zero_bytes\": %lu,\n", ref_zero_bytes );
    printf( "  \"ref_residual_bytes\": %lu,\n", ref_left );
    printf( "  \"ref_distinct_keys\": %lu,\n", distinct );
    printf( "  \"ref_entropy_bits\": %.4f,\n", H );
  }
  if( want_field ) {
    /* H over the escape frequency distribution is what a reference into a
       dictionary of already escaped keys costs; the keys themselves still
       have to be transmitted once, 32 bytes each. */
    double H = 0.0; u64 distinct = 0UL, esc = fld_esc[1];
    u64 top = 0UL, singles = 0UL, repeats = 0UL, hist[64]; memset(hist,0,sizeof(hist));
    for( u64 i=0UL;i<ef_cap;i++ ) if( ef[i].fp ) {
      distinct++;
      if( ef[i].cnt==1UL ) singles++; else repeats += ef[i].cnt-1UL;
      if( ef[i].cnt>top ) top = ef[i].cnt;
      hist[ bitlen(ef[i].cnt) ]++;
      double p = (double)ef[i].cnt/(double)(esc?esc:1UL);
      H -= (double)ef[i].cnt*log2(p);
    }
    /* The codec escapes a key in full the first time and references it by rank
       afterwards, so what the rank costs is the entropy of the *repeat*
       distribution -- each key contributing cnt-1 of them -- not of the
       occurrence distribution, whose first occurrences have already paid 32
       bytes each. */
    double Hrep = 0.0;
    if( repeats ) for( u64 i=0UL;i<ef_cap;i++ ) if( ef[i].fp && ef[i].cnt>1UL ) {
      double p = (double)(ef[i].cnt-1UL)/(double)repeats;
      Hrep -= (double)(ef[i].cnt-1UL)*log2(p);
    }
    printf( "  \"tokfield_accounts\": %lu,\n", fld_acct );
    printf( "  \"tokfield_mint_zero\": %lu,\n", fld_zero[0] );
    printf( "  \"tokfield_mint_hit\": %lu,\n", fld_hit[0] );
    printf( "  \"tokfield_mint_escape\": %lu,\n", fld_esc[0] );
    printf( "  \"tokfield_owner_zero\": %lu,\n", fld_zero[1] );
    printf( "  \"tokfield_owner_hit\": %lu,\n", fld_hit[1] );
    printf( "  \"tokfield_owner_escape\": %lu,\n", fld_esc[1] );
    printf( "  \"tokfield_owner_escape_distinct\": %lu,\n", distinct );
    printf( "  \"tokfield_owner_escape_singletons\": %lu,\n", singles );
    printf( "  \"tokfield_owner_escape_max_cnt\": %lu,\n", top );
    printf( "  \"tokfield_owner_escape_entropy_bits\": %.1f,\n", H );
    printf( "  \"tokfield_owner_escape_repeats\": %lu,\n", repeats );
    printf( "  \"tokfield_owner_escape_repeat_entropy_bits\": %.1f,\n", Hrep );
    printf( "  \"tokfield_owner_escape_cnt_hist\": [" );
    for( int i=0;i<40;i++ ) printf( "%s%lu", i?",":"", hist[i] );
    printf( "],\n" );
    printf( "  \"tokfield_owner_escape_dropped\": %lu,\n", ef_dropped );
  }
  printf( "  \"av_sample_vecs\": %lu,\n", av_vecs );
  printf( "  \"av_sample_records\": %lu,\n", av_cnt );
  printf( "  \"av_sample_bytes\": %lu,\n", av_written );
  printf( "  \"sample_records\": %lu,\n", samp_cnt );
  printf( "  \"sample_bytes\": %lu\n", samp_written );
  printf( "}\n" );
  return 0;
}
