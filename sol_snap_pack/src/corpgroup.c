/* corpgroup - extract one (owner,data_len) group from a corpus.

   usage: corpgroup corpus owner_b58 len outdir

   Writes:
     raw.bin   the records' data, concatenated (what the compressor sees today)
     col.bin   the same bytes transposed (all byte 0s, then all byte 1s, ...)
     key0.bin  the 32 byte field at offset 0 of each record
     key32.bin the 32 byte field at offset 32 of each record
   so that a generic compressor can be measured on each layout. */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef uint8_t u8; typedef uint64_t u64;
static inline u64 ld8( u8 const * p ){ u64 v; memcpy(&v,p,8); return v; }

static int
b58_dec( char const * s, u8 * out ) {
  static char const * A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  u8 acc[32]; memset(acc,0,32);
  for( char const * p=s; *p; p++ ) {
    char const * q = strchr(A,*p); if(!q) return -1;
    int carry = (int)(q-A);
    for( int i=31;i>=0;i-- ){ int t = acc[i]*58 + carry; acc[i]=(u8)t; carry=t>>8; }
    if( carry ) return -1;
  }
  memcpy(out,acc,32); return 0;
}

int
main( int argc, char ** argv ){
  if( argc<5 ){ fprintf(stderr,"usage: %s corpus owner_b58 len outdir\n",argv[0]); return 1; }
  u8 want_owner[32];
  if( b58_dec(argv[2],want_owner) ){ fprintf(stderr,"bad base58\n"); return 1; }
  u64 want_len = strtoull(argv[3],NULL,10);
  int fd = open(argv[1],O_RDONLY); if(fd<0){ perror("open"); return 1; }
  struct stat st; fstat(fd,&st); u64 fsz=(u64)st.st_size;
  u8 * base = mmap(NULL,fsz,PROT_READ,MAP_PRIVATE,fd,0);
  if(base==MAP_FAILED){ perror("mmap"); return 1; }
  madvise(base,fsz,MADV_SEQUENTIAL);

  u64 n = 0UL;
  for( u64 off=16UL; off+104UL<=fsz; ){
    u64 dlen = ld8(base+off+24);
    if( off+104UL+dlen>fsz ) break;
    if( dlen==want_len && !memcmp(base+off+64,want_owner,32) ) n++;
    off += 104UL+dlen;
  }
  fprintf(stderr,"%lu records of len %lu\n",n,want_len);
  if( !n ) return 1;

  u8 * buf = malloc(n*want_len);
  u64 i = 0UL;
  for( u64 off=16UL; off+104UL<=fsz; ){
    u64 dlen = ld8(base+off+24);
    if( off+104UL+dlen>fsz ) break;
    if( dlen==want_len && !memcmp(base+off+64,want_owner,32) ) memcpy(buf+i++*want_len,base+off+104,want_len);
    off += 104UL+dlen;
  }

  char p[4096];
  snprintf(p,sizeof(p),"%s/raw.bin",argv[4]);
  FILE * f = fopen(p,"wb"); fwrite(buf,1,n*want_len,f); fclose(f);

  u8 * col = malloc(n*want_len);
  for( u64 o=0UL;o<want_len;o++ ) for( u64 r=0UL;r<n;r++ ) col[o*n+r] = buf[r*want_len+o];
  snprintf(p,sizeof(p),"%s/col.bin",argv[4]);
  f = fopen(p,"wb"); fwrite(col,1,n*want_len,f); fclose(f);

  if( want_len>=32UL ){
    snprintf(p,sizeof(p),"%s/key0.bin",argv[4]);
    f=fopen(p,"wb"); for( u64 r=0UL;r<n;r++ ) fwrite(buf+r*want_len,1,32,f); fclose(f);
  }
  if( want_len>=64UL ){
    snprintf(p,sizeof(p),"%s/key32.bin",argv[4]);
    f=fopen(p,"wb"); for( u64 r=0UL;r<n;r++ ) fwrite(buf+r*want_len+32,1,32,f); fclose(f);
  }
  printf("%lu\n",n);
  return 0;
}
