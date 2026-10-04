/* corpstat - structural analysis of a SNAPCORP1 account corpus.

   Answers the questions a snapshot-specific codec needs:
     - how much of account *data* is really pubkeys that appear elsewhere in
       the snapshot (dictionary / back-reference potential),
     - per (owner,data_len) group, how much information each byte offset
       actually carries (per-field cardinality and order-0 entropy),
     - how skewed the owner and data_len alphabets are (ANS potential).  */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

typedef struct {
  u64 slot, lamports, rent_epoch, data_len;
  u8 const * pubkey;
  u8 const * owner;
  int exec;
  u8 const * data;
} rec_t;

static u8 const * base;
static u64        fsz;

static inline u64 ld8( u8 const * p ) { u64 v; memcpy(&v,p,8); return v; }
static inline u64 mix( u64 h ) { h ^= h>>33; h *= 0xff51afd7ed558ccdUL; h ^= h>>33; h *= 0xc4ceb9fe1a85ec53UL; h ^= h>>33; return h; }

/* iterate: returns 0 at end */
static int
next_rec( u64 * off, rec_t * r ) {
  u64 o = *off;
  if( o+104UL > fsz ) return 0;
  r->slot       = ld8( base+o    );
  r->lamports   = ld8( base+o+8  );
  r->rent_epoch = ld8( base+o+16 );
  r->data_len   = ld8( base+o+24 );
  r->pubkey     = base+o+32;
  r->owner      = base+o+64;
  r->exec       = base[o+96];
  r->data       = base+o+104;
  if( o+104UL+r->data_len > fsz ) return 0;
  *off = o+104UL+r->data_len;
  return 1;
}

/* ---------- exact 32 byte key set ------------------------------------ */

typedef struct { u8 key[32]; u32 cnt; u32 used; } k32;
static k32 * kt;
static u64   kt_cap, kt_mask, kt_cnt;

static void kt_init( u64 lg ) {
  kt_cap = 1UL<<lg; kt_mask = kt_cap-1UL;
  kt = calloc( kt_cap, sizeof(k32) );
  if( !kt ) { fprintf(stderr,"oom kt\n"); exit(1); }
}
static k32 * kt_get( u8 const * k, int insert ) {
  u64 i = mix( ld8(k) ) & kt_mask;
  for(;;) {
    k32 * e = kt+i;
    if( !e->used ) {
      if( !insert ) return NULL;
      e->used = 1; memcpy( e->key, k, 32 ); kt_cnt++;
      return e;
    }
    if( !memcmp( e->key, k, 32 ) ) return e;
    i = (i+1UL)&kt_mask;
  }
}

/* single hash bloom over 8 byte prefixes */
#define BLOOM_LG 28
static u64 * bloom;
static void bloom_add( u64 v ) { u64 h = mix(v) & ((1UL<<BLOOM_LG)-1UL); bloom[h>>6] |= 1UL<<(h&63UL); }
static inline int bloom_has( u64 v ) { u64 h = mix(v) & ((1UL<<BLOOM_LG)-1UL); return (bloom[h>>6]>>(h&63UL))&1UL; }

/* ---------- (owner,len) groups --------------------------------------- */

#define FLD_OFF_MAX 512UL
#define FLD_STEP    4UL
#define FLD_CARD_MAX 65536UL

typedef struct { u64 v, c; int used; } vc;
typedef struct {
  vc * t; u64 cap, cnt; int over;   /* over: cardinality above cap */
  u64 tot;
} card_t;

static void card_init( card_t * c ) { c->cap = 8UL; c->cnt = 0UL; c->over = 0; c->tot = 0UL; c->t = calloc(c->cap,sizeof(vc)); }
static void card_add( card_t * c, u64 v ) {
  c->tot++;
  if( c->over ) return;
  if( c->cnt*2UL >= c->cap ) {
    if( c->cap>=FLD_CARD_MAX ) { c->over = 1; return; }
    u64 ncap = c->cap*4UL;
    vc * nt = calloc( ncap, sizeof(vc) );
    for( u64 i=0UL;i<c->cap;i++ ) if( c->t[i].used ) {
      u64 j = mix(c->t[i].v)&(ncap-1UL);
      while( nt[j].used ) j = (j+1UL)&(ncap-1UL);
      nt[j] = c->t[i];
    }
    free( c->t ); c->t = nt; c->cap = ncap;
  }
  u64 i = mix(v)&(c->cap-1UL);
  for(;;) {
    if( !c->t[i].used ) { c->t[i].used=1; c->t[i].v=v; c->t[i].c=1UL; c->cnt++; return; }
    if( c->t[i].v==v ) { c->t[i].c++; return; }
    i = (i+1UL)&(c->cap-1UL);
  }
}
static double card_entropy( card_t * c ) {
  if( c->over || !c->tot ) return -1.0;
  double h = 0.0;
  for( u64 i=0UL;i<c->cap;i++ ) if( c->t[i].used ) {
    double p = (double)c->t[i].c/(double)c->tot;
    h -= p*log2(p);
  }
  return h;
}

typedef struct {
  u8   owner[32];
  u64  len, cnt, bytes;
  card_t * f8;    /* u64 field at each offset (step FLD_STEP) */
  card_t * f32;   /* 32 byte field at each offset (step FLD_STEP) */
  u64  nf;
  u64  pk_hits;   /* 32B windows in this group's data matching known pubkeys */
  card_t lam, rent, exe;   /* conditional distributions given (owner,len) */
  card_t dlen_of_owner;    /* only used on the owner-only group (len==~0) */
  int  used;
} grp_t;

static grp_t * gt;
static u64     gt_cap, gt_mask, gt_cnt;

static void gt_init( u64 lg ) { gt_cap = 1UL<<lg; gt_mask = gt_cap-1UL; gt = calloc(gt_cap,sizeof(grp_t)); }
static grp_t * gt_get( u8 const * owner, u64 len ) {
  u64 i = mix( ld8(owner)^(len*0x9E3779B97F4A7C15UL) )&gt_mask;
  for(;;) {
    grp_t * g = gt+i;
    if( !g->used ) { g->used=1; memcpy(g->owner,owner,32); g->len=len; gt_cnt++; return g; }
    if( g->len==len && !memcmp(g->owner,owner,32) ) return g;
    i = (i+1UL)&gt_mask;
  }
}
static int cmp_grp( void const * a, void const * b ) {
  u64 x=(*(grp_t * const *)a)->cnt, y=(*(grp_t * const *)b)->cnt;
  return x<y?1:(x>y?-1:0);
}
static int cmp_k32( void const * a, void const * b ) {
  u32 x=(*(k32 * const *)a)->cnt, y=(*(k32 * const *)b)->cnt;
  return x<y?1:(x>y?-1:0);
}

static void
b58( u8 const * k, char * out ) {
  static char const * A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  u8 tmp[32]; memcpy(tmp,k,32);
  char buf[64]; int n=0;
  int zeros=0; while( zeros<32 && !tmp[zeros] ) zeros++;
  int start=zeros;
  while( start<32 ) {
    int rem=0, nz=-1;
    for( int i=start;i<32;i++ ) {
      int cur = (rem<<8)|tmp[i];
      tmp[i] = (u8)(cur/58); rem = cur%58;
      if( nz<0 && tmp[i] ) nz=i;
    }
    buf[n++] = A[rem];
    start = nz<0 ? 32 : nz;
  }
  int o=0;
  for( int i=0;i<zeros;i++ ) out[o++]='1';
  while( n ) out[o++] = buf[--n];
  out[o]='\0';
}

/* ---------- repeated 32 byte window dictionary ----------------------- */
/* Counts every 4 byte aligned 32 byte window of account data by 64 bit hash,
   then greedily covers the data with windows that occur more than once.  This
   is the "how much of the data is a repeated pubkey sized blob" measurement,
   independent of whether that blob is an account in this corpus. */

typedef struct { u64 h; u32 cnt; } w_ent;
static w_ent * wt;
static u64     wt_cap, wt_mask, wt_cnt;

static void wt_init( u64 lg ) {
  wt_cap = 1UL<<lg; wt_mask = wt_cap-1UL;
  wt = calloc( wt_cap, sizeof(w_ent) );
  if( !wt ) { fprintf(stderr,"oom wt\n"); exit(1); }
}
static inline u64 win_hash( u8 const * p ) {
  return mix( ld8(p)*0x9E3779B97F4A7C15UL + ld8(p+8)*0xC2B2AE3D27D4EB4FUL +
              ld8(p+16)*0x165667B19E3779F9UL + ld8(p+24)*0x27D4EB2F165667C5UL );
}
static w_ent * wt_get( u64 h, int insert ) {
  u64 i = h & wt_mask;
  for(;;) {
    w_ent * e = wt+i;
    if( !e->cnt ) {
      if( !insert ) return NULL;
      if( wt_cnt*4UL > wt_cap*3UL ) return NULL;
      e->h = h; wt_cnt++; return e;
    }
    if( e->h==h ) return e;
    i = (i+1UL)&wt_mask;
  }
}

int
main( int argc, char ** argv ) {
  if( argc<2 ) { fprintf(stderr,"usage: %s corpus [top_groups]\n",argv[0]); return 1; }
  u64 topn = argc>2 ? strtoull(argv[2],NULL,10) : 24UL;
  int fd = open( argv[1], O_RDONLY );
  if( fd<0 ) { perror("open"); return 1; }
  struct stat st; fstat( fd, &st ); fsz = (u64)st.st_size;
  u8 * m = mmap( NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0 );
  if( m==MAP_FAILED ) { perror("mmap"); return 1; }
  madvise( m, fsz, MADV_SEQUENTIAL );
  if( memcmp( m, "SNAPCORP1", 9 ) ) { fprintf(stderr,"bad magic\n"); return 1; }
  base = m; /* records start at 16 */

  kt_init( 26 );
  gt_init( 20 );
  bloom = calloc( 1UL<<(BLOOM_LG-6), 8UL );

  /* pass 1: record level stats, key universe, groups */
  u64 off = 16UL, n = 0UL, data_bytes = 0UL;
  u64 rent_exempt_exact = 0UL, rent_exempt_ge = 0UL;
  u64 slot_prev = 0UL, slot_back = 0UL, slot_delta_bits = 0UL;
  rec_t r;
  while( next_rec( &off, &r ) ) {
    n++; data_bytes += r.data_len;
    k32 * e = kt_get( r.pubkey, 1 ); e->cnt++;
    bloom_add( ld8( r.pubkey ) );
    k32 * oe = kt_get( r.owner, 1 ); oe->cnt++;
    bloom_add( ld8( r.owner ) );
    if( n>1 ) { if( r.slot<slot_prev ) slot_back++; }
    slot_prev = r.slot;
    grp_t * g = gt_get( r.owner, r.data_len );
    if( !g->cnt ) { card_init( &g->lam ); card_init( &g->rent ); card_init( &g->exe ); }
    g->cnt++; g->bytes += r.data_len;
    card_add( &g->lam, r.lamports );
    card_add( &g->rent, r.rent_epoch );
    card_add( &g->exe, (u64)r.exec );
    grp_t * go = gt_get( r.owner, (u64)-1 );      /* owner only bucket */
    if( !go->cnt ) card_init( &go->dlen_of_owner );
    go->cnt++;
    card_add( &go->dlen_of_owner, r.data_len );
    if( r.lamports == (128UL+r.data_len)*6960UL ) rent_exempt_exact++;
    if( r.lamports >= (128UL+r.data_len)*6960UL ) rent_exempt_ge++;
  }
  (void)slot_delta_bits;

  /* pick the groups that get a field profile */
  u64 k = 0UL;
  grp_t ** gv = malloc( gt_cnt*sizeof(grp_t*) );
  for( u64 i=0UL;i<gt_cap;i++ ) if( gt[i].used ) gv[k++] = gt+i;
  qsort( gv, k, sizeof(grp_t*), cmp_grp );
  u64 ng = k<topn ? k : topn;
  for( u64 i=0UL;i<ng;i++ ) {
    grp_t * g = gv[i];
    if( !g->len ) continue;
    g->nf = (g->len+FLD_STEP-1UL)/FLD_STEP;
    if( g->nf*FLD_STEP > FLD_OFF_MAX ) g->nf = FLD_OFF_MAX/FLD_STEP;
    g->f8  = calloc( g->nf, sizeof(card_t) );
    g->f32 = calloc( g->nf, sizeof(card_t) );
    for( u64 j=0UL;j<g->nf;j++ ) { card_init( g->f8+j ); card_init( g->f32+j ); }
  }

  /* pass 2: field profiles + pubkey back-reference scan */
  u64 pk_hit_bytes = 0UL, pk_hit_cnt = 0UL, bloom_probe = 0UL, bloom_fp = 0UL;
  u64 self_ref = 0UL, zero32_cnt = 0UL, zero32_bytes = 0UL;
  u8  zerokey[32]; memset( zerokey, 0, 32 );
  off = 16UL;
  while( next_rec( &off, &r ) ) {
    grp_t * g = gt_get( r.owner, r.data_len );
    if( g->f8 ) {
      for( u64 j=0UL;j<g->nf;j++ ) {
        u64 o = j*FLD_STEP;
        if( o+8UL<=r.data_len ) card_add( g->f8+j, ld8(r.data+o) );
        if( o+32UL<=r.data_len ) {
          u64 h = mix(ld8(r.data+o))^mix(ld8(r.data+o+8))^mix(ld8(r.data+o+16))^mix(ld8(r.data+o+24));
          card_add( g->f32+j, h );
        }
      }
    }
    /* scan for 32 byte windows that are pubkeys known to the snapshot */
    if( r.data_len>=32UL ) {
      u64 i = 0UL;
      while( i+32UL<=r.data_len ) {
        u64 v = ld8( r.data+i );
        bloom_probe++;
        if( !bloom_has( v ) ) { i++; continue; }
        if( !v && !memcmp( r.data+i, zerokey, 32 ) ) {
          /* an all zero window is not a real reference, it is just zeros */
          zero32_cnt++; zero32_bytes += 32UL; i += 32UL; continue;
        }
        k32 * e = kt_get( r.data+i, 0 );
        if( !e ) { bloom_fp++; i++; continue; }
        pk_hit_cnt++; pk_hit_bytes += 32UL;
        e->cnt++;
        if( !memcmp( r.data+i, r.pubkey, 32 ) ) self_ref++;
        i += 32UL;
      }
    }
  }

  printf( "corpus: %s\n", argv[1] );
  printf( "records            : %lu\n", n );
  printf( "data bytes         : %lu (%.2f MiB)\n", data_bytes, (double)data_bytes/1048576.0 );
  printf( "meta bytes (104/rec): %lu\n", n*104UL );
  printf( "distinct 32B keys  : %lu\n", kt_cnt );
  printf( "slot goes backwards: %lu times\n", slot_back );
  printf( "(owner,len) groups : %lu\n", gt_cnt );
  {
    /* H(owner), H(dlen|owner), H(lamports|owner,dlen), H(rent|owner,dlen),
       H(exec|owner,dlen): the cost of the per account metadata under an
       order-0 model with the obvious contexts. */
    double h_owner = 0.0, h_dlen = 0.0, h_lam = 0.0, h_rent = 0.0, h_exe = 0.0;
    u64 owner_tot = 0UL;
    for( u64 i=0UL;i<gt_cap;i++ ) if( gt[i].used && gt[i].len==(u64)-1 ) owner_tot += gt[i].cnt;
    for( u64 i=0UL;i<gt_cap;i++ ) {
      grp_t * g = gt+i;
      if( !g->used ) continue;
      if( g->len==(u64)-1 ) {
        double p = (double)g->cnt/(double)owner_tot;
        h_owner -= p*log2(p);
        double h = card_entropy( &g->dlen_of_owner );
        if( h>=0.0 ) h_dlen += p*h;
        continue;
      }
      double w = (double)g->cnt/(double)n;
      double a = card_entropy( &g->lam ), b = card_entropy( &g->rent ), c = card_entropy( &g->exe );
      h_lam  += w*(a>=0.0?a:16.0);
      h_rent += w*(b>=0.0?b:2.0);
      h_exe  += w*(c>=0.0?c:1.0);
    }
    printf( "\nmetadata cost under simple conditional models (bits per account):\n" );
    printf( "  H(owner)                 %6.2f bits\n", h_owner );
    printf( "  H(data_len | owner)      %6.2f bits\n", h_dlen );
    printf( "  H(lamports | owner,len)  %6.2f bits\n", h_lam );
    printf( "  H(rent_epoch| owner,len) %6.2f bits\n", h_rent );
    printf( "  H(executable| owner,len) %6.2f bits\n", h_exe );
    printf( "  pubkey (incompressible)  %6.2f bits\n", 256.0 - log2((double)(n?n:1)) + 1.44 );
    printf( "  ---> metadata total      %6.2f bits = %.1f bytes/account (vs 136 raw)\n",
            h_owner+h_dlen+h_lam+h_rent+h_exe+256.0-log2((double)(n?n:1))+1.44,
            (h_owner+h_dlen+h_lam+h_rent+h_exe+256.0-log2((double)(n?n:1))+1.44)/8.0 );
    printf( "  lamports == rent exempt minimum: %lu (%.1f%%), >= minimum: %lu (%.1f%%)\n",
            rent_exempt_exact, 100.0*(double)rent_exempt_exact/(double)n,
            rent_exempt_ge, 100.0*(double)rent_exempt_ge/(double)n );
  }

  printf( "\npubkey back-references inside account data (non overlapping, any offset):\n" );
  printf( "  windows probed   : %lu\n", bloom_probe );
  printf( "  bloom false pos  : %lu (%.4f%%)\n", bloom_fp, 100.0*(double)bloom_fp/(double)(bloom_probe?bloom_probe:1) );
  printf( "  matches          : %lu  covering %lu bytes = %.2f%% of data\n",
          pk_hit_cnt, pk_hit_bytes, 100.0*(double)pk_hit_bytes/(double)(data_bytes?data_bytes:1) );
  printf( "  self references  : %lu\n", self_ref );
  printf( "  all-zero windows : %lu covering %lu bytes = %.2f%% of data\n",
          zero32_cnt, zero32_bytes, 100.0*(double)zero32_bytes/(double)(data_bytes?data_bytes:1) );
  printf( "  residual (not zero, not a known key): %.2f%% of data\n",
          100.0*(double)(data_bytes-zero32_bytes-pk_hit_bytes)/(double)(data_bytes?data_bytes:1) );

  /* most referenced keys overall */
  {
    u64 kk = 0UL;
    k32 ** v = malloc( kt_cnt*sizeof(k32*) );
    for( u64 i=0UL;i<kt_cap;i++ ) if( kt[i].used ) { int nz=0; for(int j=0;j<32;j++) nz|=kt[i].key[j]; if( nz ) v[kk++] = kt+i; }
    qsort( v, kk, sizeof(k32*), cmp_k32 );
    printf( "\ntop referenced 32B keys:\n" );
    char s[64];
    for( u64 i=0UL;i<32UL && i<kk;i++ ) { b58( v[i]->key, s ); printf( "  %-46s %u\n", s, v[i]->cnt ); }
    /* how concentrated: coverage by top N */
    u64 tot = 0UL; for( u64 i=0UL;i<kk;i++ ) tot += v[i]->cnt;
    u64 acc = 0UL;
    printf( "  coverage: " );
    u64 marks[] = { 16UL, 256UL, 4096UL, 65536UL, 1048576UL };
    u64 mi = 0UL;
    for( u64 i=0UL;i<kk && mi<5UL;i++ ) {
      acc += v[i]->cnt;
      if( i+1UL==marks[mi] ) { printf( "top%lu=%.1f%% ", marks[mi], 100.0*(double)acc/(double)tot ); mi++; }
    }
    printf( "\n" );
    free( v );
  }

  /* pass 3: repeated window dictionary */
  {
    wt_init( 28 );
    u8 zerokey2[32]; memset( zerokey2, 0, 32 );
    off = 16UL;
    while( next_rec( &off, &r ) ) {
      for( u64 i=0UL; i+32UL<=r.data_len; i+=4UL ) {
        if( !ld8(r.data+i) && !memcmp( r.data+i, zerokey2, 32 ) ) continue;
        w_ent * e = wt_get( win_hash( r.data+i ), 1 );
        if( e ) e->cnt++;
      }
    }
    u64 cov_bytes = 0UL, cov_cnt = 0UL, zcov = 0UL, uncov = 0UL;
    off = 16UL;
    while( next_rec( &off, &r ) ) {
      u64 i = 0UL;
      while( i+32UL<=r.data_len ) {
        if( !ld8(r.data+i) && !memcmp( r.data+i, zerokey2, 32 ) ) { zcov += 32UL; i += 32UL; continue; }
        w_ent * e = wt_get( win_hash( r.data+i ), 0 );
        if( e && e->cnt>1U ) { cov_bytes += 32UL; cov_cnt++; i += 32UL; continue; }
        uncov += 4UL; i += 4UL;
      }
      uncov += r.data_len>=32UL ? (r.data_len-i) : r.data_len;
    }
    printf( "\nrepeated 32B window dictionary (4 byte aligned windows, hash counted):\n" );
    printf( "  distinct windows : %lu%s\n", wt_cnt, wt_cnt*4UL>wt_cap*3UL?" (TABLE FULL, undercount)":"" );
    printf( "  repeated windows cover %lu bytes = %.2f%% of data (%lu refs)\n",
            cov_bytes, 100.0*(double)cov_bytes/(double)(data_bytes?data_bytes:1), cov_cnt );
    printf( "  zero windows cover     %lu bytes = %.2f%%\n", zcov, 100.0*(double)zcov/(double)(data_bytes?data_bytes:1) );
    printf( "  left over              %lu bytes = %.2f%%\n", uncov, 100.0*(double)uncov/(double)(data_bytes?data_bytes:1) );
  }

  printf( "\nper group field profile (offset step %lu, '8B' = u64 field, '32B' = pubkey sized field)\n", FLD_STEP );
  for( u64 i=0UL;i<ng;i++ ) {
    grp_t * g = gv[i];
    char s[64]; b58( g->owner, s );
    printf( "\n[%lu] owner=%s len=%lu cnt=%lu\n", i, s, g->len, g->cnt );
    if( !g->f8 ) continue;
    double tot8 = 0.0;
    printf( "  off  card8   H8(bits)  card32  H32(bits)\n" );
    for( u64 j=0UL;j<g->nf;j++ ) {
      card_t * a = g->f8+j; card_t * b = g->f32+j;
      double h8 = card_entropy(a), h32 = card_entropy(b);
      if( h8>=0.0 ) tot8 += h8;
      if( !a->tot && !b->tot ) continue;
      printf( "  %3lu  %6s  %8s  %6s  %8s\n", j*FLD_STEP,
              a->over?">64k":(a->tot?({ static char bb[16]; snprintf(bb,16,"%lu",a->cnt); bb; }):"-"),
              h8>=0.0?({ static char bb[16]; snprintf(bb,16,"%.2f",h8); bb; }):"?",
              b->over?">64k":(b->tot?({ static char bb[16]; snprintf(bb,16,"%lu",b->cnt); bb; }):"-"),
              h32>=0.0?({ static char bb[16]; snprintf(bb,16,"%.2f",h32); bb; }):"?" );
    }
    printf( "  sum of per-offset u64 entropies (step %lu): %.1f bits over %lu bytes\n", FLD_STEP, tot8, g->len );
  }
  return 0;
}
