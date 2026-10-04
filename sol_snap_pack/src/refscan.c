/* refscan - how much of account data is a reference to a pubkey that the
   snapshot already carries?

   usage: refscan keys.bin corpus [step]

   keys.bin is the raw dump of every account pubkey in the snapshot
   (snapscan -k), corpus is a SNAPCORP1 sample.  Every `step` aligned 32 byte
   window of account data is looked up in the key set; on a hit the window is
   consumed (32 bytes) and the hit recorded, otherwise the scan advances by
   `step`.

   The key set is stored as 64 bit fingerprints in an open addressed table, so
   a "hit" can in principle be a fingerprint collision; at 2^64 fingerprints
   and ~1e9 probes the expected number of false hits is well under one. */

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

static inline u64 ld8( u8 const * p ){ u64 v; memcpy(&v,p,8); return v; }
static inline u64 mix( u64 h ){ h^=h>>33; h*=0xff51afd7ed558ccdUL; h^=h>>33; h*=0xc4ceb9fe1a85ec53UL; h^=h>>33; return h; }
static inline u64 key_fp( u8 const * p ){
  u64 h = ld8(p)*0x9E3779B97F4A7C15UL + ld8(p+8)*0xC2B2AE3D27D4EB4FUL +
          ld8(p+16)*0x165667B19E3779F9UL + ld8(p+24)*0x27D4EB2F165667C5UL;
  h = mix(h);
  return h ? h : 1UL;   /* 0 marks an empty slot */
}

static u64 * tbl;
static u64   tcap, tmask, tcnt;
static char const * hot_path = NULL;
static int cmp_hot( void const * a, void const * b ){
  u64 const * x=a, * y=b;               /* descending by count */
  return x[1]<y[1] ? 1 : (x[1]>y[1] ? -1 : 0);
}

static void tbl_init( u64 lg ){
  tcap = 1UL<<lg; tmask = tcap-1UL;
  tbl = mmap( NULL, tcap*8UL, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0 );
  if( tbl==MAP_FAILED ){ perror("mmap tbl"); exit(1); }
  madvise( tbl, tcap*8UL, MADV_HUGEPAGE );
}
static void tbl_add( u64 fp ){
  u64 i = fp & tmask;
  for(;;){
    if( !tbl[i] ){ tbl[i]=fp; tcnt++; return; }
    if( tbl[i]==fp ) return;
    i = (i+1UL)&tmask;
  }
}
static inline int tbl_has( u64 fp ){
  u64 i = fp & tmask;
  for(;;){
    u64 v = tbl[i];
    if( !v ) return 0;
    if( v==fp ) return 1;
    i = (i+1UL)&tmask;
  }
}

static int cmp_u64( void const * a, void const * b ){
  u64 x=*(u64 const *)a, y=*(u64 const *)b;
  return x<y?-1:(x>y?1:0);
}

int
main( int argc, char ** argv ){
  if( argc<3 ){ fprintf(stderr,"usage: %s keys.bin corpus [step] [-H hot.bin]\n",argv[0]); return 1; }
  u64 step = (argc>3 && argv[3][0]!='-') ? strtoull(argv[3],NULL,10) : 4UL;
  for( int i=3; i<argc; i++ )
    if( !strcmp(argv[i],"-H") && i+1<argc ) hot_path = argv[++i];

  /* load the key set */
  int kfd = open(argv[1],O_RDONLY); if(kfd<0){ perror("open keys"); return 1; }
  struct stat ks; fstat(kfd,&ks);
  u64 nkeys = (u64)ks.st_size/32UL;
  u64 lg = 20UL; while( (1UL<<lg) < nkeys*2UL ) lg++;
  fprintf(stderr,"keys: %lu, table 2^%lu (%.1f GiB)\n", nkeys, lg, (double)(1UL<<lg)*8.0/1073741824.0);
  tbl_init( lg );
  {
    size_t bufsz = 32UL<<20;
    u8 * buf = malloc(bufsz);
    u64 done = 0UL;
    for(;;){
      ssize_t n = read( kfd, buf, bufsz );
      if( n<0 ){ perror("read keys"); return 1; }
      if( !n ) break;
      for( ssize_t i=0; i+32<=n; i+=32 ) tbl_add( key_fp(buf+i) );
      done += (u64)n/32UL;
      if( !(done % (1UL<<26)) ) fprintf(stderr,"\r  loaded %lu keys", done), fflush(stderr);
    }
    free(buf);
    fprintf(stderr,"\r  loaded %lu keys, %lu distinct fingerprints\n", done, tcnt);
  }
  close(kfd);

  /* scan the corpus */
  int fd = open(argv[2],O_RDONLY); if(fd<0){ perror("open corpus"); return 1; }
  struct stat st; fstat(fd,&st);
  u64 fsz=(u64)st.st_size;
  u8 * base = mmap(NULL,fsz,PROT_READ,MAP_PRIVATE,fd,0);
  if(base==MAP_FAILED){ perror("mmap corpus"); return 1; }
  madvise(base,fsz,MADV_SEQUENTIAL);
  if( memcmp(base,"SNAPCORP1",9) ){ fprintf(stderr,"bad magic\n"); return 1; }

  u64 hit_cap = 1UL<<24, hit_n = 0UL;
  u64 * hits = malloc(hit_cap*8UL);
  u64 n=0UL, data_bytes=0UL, probes=0UL, hit_bytes=0UL, zero_bytes=0UL, left=0UL, self=0UL, owner_hit=0UL;
  u8 zero32[32]; memset(zero32,0,32);
  for( u64 off=16UL; off+104UL<=fsz; ){
    u64 dlen = ld8(base+off+24);
    if( off+104UL+dlen>fsz ) break;
    u8 const * pk = base+off+32;
    u8 const * ow = base+off+64;
    u8 const * d  = base+off+104;
    n++; data_bytes += dlen;
    u64 i=0UL;
    while( i+32UL<=dlen ){
      if( !ld8(d+i) && !memcmp(d+i,zero32,32) ){ zero_bytes += 32UL; i += 32UL; continue; }
      probes++;
      u64 fp = key_fp(d+i);
      if( tbl_has(fp) ){
        hit_bytes += 32UL;
        if( hit_n==hit_cap ){ hit_cap*=2UL; hits=realloc(hits,hit_cap*8UL); }
        hits[hit_n++]=fp;
        if( !memcmp(d+i,pk,32) ) self++;
        if( !memcmp(d+i,ow,32) ) owner_hit++;
        i += 32UL;
      } else {
        left += step; i += step;
      }
    }
    left += dlen-i;
    off += 104UL+dlen;
  }

  printf("corpus records      : %lu\n", n);
  printf("data bytes          : %lu (%.2f MiB)\n", data_bytes, (double)data_bytes/1048576.0);
  printf("window step         : %lu\n", step);
  printf("probes              : %lu\n", probes);
  printf("zero windows        : %lu bytes = %.2f%%\n", zero_bytes, 100.0*(double)zero_bytes/(double)data_bytes);
  printf("KEY references      : %lu (%lu bytes) = %.2f%% of data\n", hit_n, hit_bytes, 100.0*(double)hit_bytes/(double)data_bytes);
  printf("  of which self     : %lu\n", self);
  printf("  of which owner    : %lu\n", owner_hit);
  printf("residual            : %lu bytes = %.2f%%\n", left, 100.0*(double)left/(double)data_bytes);
  printf("references/account  : %.2f\n", (double)hit_n/(double)n);

  /* frequency distribution of the referenced keys */
  qsort( hits, hit_n, 8UL, cmp_u64 );
  /* -H FILE: dump (fingerprint,count) for every referenced key, most
     referenced first.  This is the candidate hot table a codec would ship in
     its preamble (or inherit from the previous snapshot). */
  if( hot_path ) {
    u64 m=0UL; for( u64 i=0UL;i<hit_n; ){ u64 j=i; while(j<hit_n&&hits[j]==hits[i]) j++; m++; i=j; }
    u64 (*hc)[2] = malloc(m*16UL); u64 k=0UL;
    for( u64 i=0UL;i<hit_n; ){ u64 j=i; while(j<hit_n&&hits[j]==hits[i]) j++;
                               hc[k][0]=hits[i]; hc[k][1]=j-i; k++; i=j; }
    qsort( hc, m, 16UL, cmp_hot );
    FILE * hf = fopen( hot_path, "wb" );
    if( !hf ) { perror("open hot"); return 1; }
    fwrite( hc, 16UL, m, hf ); fclose( hf );
    fprintf( stderr, "hot table: %lu keys -> %s\n", m, hot_path );
  }
  u64 distinct=0UL;
  double H=0.0;
  u64 * freq = malloc(hit_n*8UL); u64 fn=0UL;
  for( u64 i=0UL;i<hit_n; ){
    u64 j=i; while( j<hit_n && hits[j]==hits[i] ) j++;
    u64 c=j-i; distinct++;
    freq[fn++]=c;
    double p=(double)c/(double)hit_n;
    H -= p*log2(p);
    i=j;
  }
  qsort( freq, fn, 8UL, cmp_u64 );
  printf("distinct keys refd  : %lu\n", distinct);
  printf("H(reference)        : %.2f bits  -> %.2f bytes per reference\n", H, H/8.0);
  printf("log2(distinct)      : %.2f bits\n", log2((double)(distinct?distinct:1)));
  /* top N coverage, freq[] is ascending */
  u64 marks[]={16UL,256UL,4096UL,65536UL,1048576UL};
  for( int mi=0; mi<5; mi++ ){
    u64 m=marks[mi]; if( m>fn ) m=fn;
    u64 acc=0UL; for( u64 i=0UL;i<m;i++ ) acc += freq[fn-1UL-i];
    printf("  top %-8lu covers %.1f%% of references\n", marks[mi], 100.0*(double)acc/(double)hit_n);
  }
  return 0;
}
