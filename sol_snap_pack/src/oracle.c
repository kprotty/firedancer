/* oracle - price a SNAPCORP1 corpus under successively richer models,
   without implementing a codec (PLAN.md Phase 1 step 2).

   usage: oracle keys.bin corpus outdir [topN] [maxlen]

   keys.bin is the full pubkey dictionary (snapscan -k), corpus a sample.
   Models, in the same order as PLAN.md:

     M1  field split metadata (owner/len/lamports/rent/exec), data untouched
         (its cost is whatever zstd gets on the raw data -- see corpsplit).
     M2  M1 + data tokenized into ZERO32 / KEYREF / LITERAL, no schema.
     M3  M2 + per (owner,data_len) templates: constant bytes, pubkey-shaped
         fields (KEYREF or literal escape), small-alphabet u64 fields.
     M4  M3 + sorted (owner,len,pubkey) order, every record XORed against
         the previous record of the same group before tokenizing -- this
         needs no template awareness, since XOR already cancels constant
         and repeated-pubkey fields down to a ZERO32 hit.
     M5  M3 + hand written layouts for the protocol-stable programs (SPL
         Token, Token-2022 + TLV extensions, stake, vote, ALT, nonce,
         Metaplex metadata), dispatched on `owner` and data_len instead of
         guessed from 8 byte aligned statistics.  Layout covered records are
         priced field by field, everything else keeps M3.

   The oracle never invents bytes: literal streams are written to outdir as
   m2_literal.bin / m3_literal.bin / m4_literal.bin so bench_baseline.sh can
   report their *real* zstd size, and everything else is priced from the
   corpus's own measured empirical distribution (order-0, honest about the
   cost of the type decision itself, not just the payload). */

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
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

static inline u64 ld8( u8 const * p ){ u64 v; memcpy(&v,p,8); return v; }
static inline u64 mix( u64 h ){ h^=h>>33; h*=0xff51afd7ed558ccdUL; h^=h>>33; h*=0xc4ceb9fe1a85ec53UL; h^=h>>33; return h; }
static inline u64 key_fp( u8 const * p ){
  u64 h = ld8(p)*0x9E3779B97F4A7C15UL + ld8(p+8)*0xC2B2AE3D27D4EB4FUL +
          ld8(p+16)*0x165667B19E3779F9UL + ld8(p+24)*0x27D4EB2F165667C5UL;
  h = mix(h);
  return h ? h : 1UL;
}

/* ---------- corpus iteration (SNAPCORP1) ------------------------------ */

typedef struct {
  u64 slot, lamports, rent_epoch, data_len;
  u8 const * pubkey; u8 const * owner; int exec; u8 const * data;
} rec_t;

static u8 const * base;
static u64        fsz;

static int
next_rec( u64 * off, rec_t * r ){
  u64 o = *off;
  if( o+104UL > fsz ) return 0;
  r->slot=ld8(base+o); r->lamports=ld8(base+o+8); r->rent_epoch=ld8(base+o+16);
  r->data_len=ld8(base+o+24); r->pubkey=base+o+32; r->owner=base+o+64;
  r->exec=base[o+96]; r->data=base+o+104;
  if( o+104UL+r->data_len > fsz ) return 0;
  *off = o+104UL+r->data_len;
  return 1;
}

/* ---------- dictionary: 64 bit fingerprints of every snapshot pubkey -- */

static u64 * dict; static u64 dict_cap, dict_mask;
static void dict_init( u64 lg ){
  dict_cap=1UL<<lg; dict_mask=dict_cap-1UL;
  dict = mmap(NULL,dict_cap*8UL,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
  if( dict==MAP_FAILED ){ perror("mmap dict"); exit(1); }
  madvise(dict,dict_cap*8UL,MADV_HUGEPAGE);
}
static void dict_add( u64 fp ){
  u64 i=fp&dict_mask;
  for(;;){ if(!dict[i]){dict[i]=fp;return;} if(dict[i]==fp) return; i=(i+1UL)&dict_mask; }
}
static inline int dict_has( u64 fp ){
  u64 i=fp&dict_mask;
  for(;;){ u64 v=dict[i]; if(!v) return 0; if(v==fp) return 1; i=(i+1UL)&dict_mask; }
}

/* ---------- generic categorical cardinality / entropy ----------------- */

#define CARD_MAX 65536UL
typedef struct { u64 v, c; int used; } vc_t;
typedef struct { vc_t * t; u64 cap, cnt, tot; int over; } card_t;

static void card_init( card_t * c ){ c->cap=8UL; c->cnt=0UL; c->tot=0UL; c->over=0; c->t=calloc(c->cap,sizeof(vc_t)); }
static void card_add( card_t * c, u64 v ){
  c->tot++;
  if( c->over ) return;
  if( c->cnt*2UL >= c->cap ){
    if( c->cap>=CARD_MAX ){ c->over=1; return; }
    u64 ncap=c->cap*4UL; vc_t * nt=calloc(ncap,sizeof(vc_t));
    for( u64 i=0UL;i<c->cap;i++ ) if( c->t[i].used ){
      u64 j=mix(c->t[i].v)&(ncap-1UL);
      while( nt[j].used ) j=(j+1UL)&(ncap-1UL);
      nt[j]=c->t[i];
    }
    free(c->t); c->t=nt; c->cap=ncap;
  }
  u64 i=mix(v)&(c->cap-1UL);
  for(;;){
    if( !c->t[i].used ){ c->t[i].used=1; c->t[i].v=v; c->t[i].c=1UL; c->cnt++; return; }
    if( c->t[i].v==v ){ c->t[i].c++; return; }
    i=(i+1UL)&(c->cap-1UL);
  }
}
static double card_entropy( card_t const * c ){
  if( c->over || !c->tot ) return -1.0;
  double h=0.0;
  for( u64 i=0UL;i<c->cap;i++ ) if( c->t[i].used ){ double p=(double)c->t[i].c/(double)c->tot; h -= p*log2(p); }
  return h;
}

/* categorical entropy of the {zero,repeat,keyref,miss} decision stream.
   `repeat` is only ever non-zero for models that compare against the
   previous record of the same group (M4b). */
typedef struct { u64 zero, repeat, keyref, miss; } dec_t;
static double dec_bits( dec_t const * d ){
  u64 tot = d->zero+d->repeat+d->keyref+d->miss;
  if( !tot ) return 0.0;
  double h=0.0, p;
  if( d->zero   ){ p=(double)d->zero/(double)tot;   h -= (double)d->zero  *log2(p); }
  if( d->repeat ){ p=(double)d->repeat/(double)tot; h -= (double)d->repeat*log2(p); }
  if( d->keyref ){ p=(double)d->keyref/(double)tot; h -= (double)d->keyref*log2(p); }
  if( d->miss   ){ p=(double)d->miss/(double)tot;   h -= (double)d->miss  *log2(p); }
  return h; /* already total bits, not per symbol */
}

/* ---------- referenced-key frequency (for KEYREF rank entropy) -------- */

static int cmp_u64( void const * a, void const * b ){ u64 x=*(u64 const*)a,y=*(u64 const*)b; return x<y?-1:(x>y?1:0); }
static double kref_entropy_bits( u64 * v, u64 n ){ /* total bits, not per symbol */
  if( !n ) return 0.0;
  qsort(v,n,8UL,cmp_u64);
  double h=0.0;
  for( u64 i=0UL;i<n; ){
    u64 j=i; while( j<n && v[j]==v[i] ) j++;
    double p=(double)(j-i)/(double)n;
    h -= (double)(j-i)*log2(p);
    i=j;
  }
  return h;
}

/* ---------- (owner,data_len) groups ------------------------------------ */

#define FLD8 8UL          /* slot width for templates */
#define PUBKEY_THRESH 0.90

enum { FT_FREE=0, FT_CONST, FT_PUBKEY, FT_PUBKEY_CONT, FT_U64 };

typedef struct {
  u8   ftype;
  u8   const_ok;         /* still constant so far (pass 2 scratch) */
  u8   const_val[8];
  card_t card;           /* raw value frequency, only meaningful if FT_U64 */
  card_t cardD;          /* XOR delta vs the previous same-group record (M4b) */
  u64  hit, tot;          /* pubkey window hit rate, only at span starts */
  u64  m5_used;          /* records that priced this slot through the template under M5 */
  u64  nb[9], nbtot;     /* significant byte counts, for the varint cross check */
} slot_t;

/* varint cost of a template slot: an ANS coded significant byte count plus a
   raw payload.  Unlike an order-0 alphabet this does not shrink just because
   the corpus is a 1/128 sample, so it is the honest price of a slot whose
   values are near unique here. */
static double
slot_varint_bits( slot_t const * s, u64 cnt ){
  double hn=0.0;
  if( cnt ) for( int i=0;i<9;i++ ) if( s->nb[i] ){
    double p=(double)s->nb[i]/(double)cnt; hn -= (double)s->nb[i]*log2(p);
  }
  return hn + 8.0*(double)s->nbtot;
}

typedef struct { u64 off, len; } span_t; /* a contiguous FT_FREE byte range, incl. tail */

typedef struct {
  u8   owner[32]; u64 len;
  u64  cnt;
  card_t lam, rent, exe;      /* M1 metadata: H(x | owner,len) */
  card_t dlen_of_owner;       /* only used on the owner-only bucket (len==~0) */
  /* M3 template, only for groups selected as top-N with len<=maxlen */
  slot_t * field; u64 nslots;
  span_t * spans; u64 nspans;
  u64  const_bytes, u64_bytes, pubkey_bytes; /* for the summary table */
  void * lay;                 /* layout_t* if a hand written layout covers this group (M5) */
  u64  fb_raw, fb_zero, fb_kref, fb_lit;  /* what this group costs in M5's fallback */
  int  used, templated, tmpl_seen;
} grp_t;

static grp_t * gt; static u64 gt_cap, gt_mask, gt_cnt;
static void gt_init( u64 lg ){ gt_cap=1UL<<lg; gt_mask=gt_cap-1UL; gt=calloc(gt_cap,sizeof(grp_t)); }
static grp_t * gt_get( u8 const * owner, u64 len ){
  u64 i = mix( ld8(owner)^(len*0x9E3779B97F4A7C15UL) )&gt_mask;
  for(;;){
    grp_t * g=gt+i;
    if( !g->used ){ g->used=1; memcpy(g->owner,owner,32); g->len=len; gt_cnt++; return g; }
    if( g->len==len && !memcmp(g->owner,owner,32) ) return g;
    i=(i+1UL)&gt_mask;
  }
}
static int cmp_grp_cnt( void const * a, void const * b ){
  u64 x=(*(grp_t*const*)a)->cnt, y=(*(grp_t*const*)b)->cnt;
  return x<y?1:(x>y?-1:0);
}

/* M4 sort: cluster records by (owner, data_len, pubkey) -- same comparator
   shape as corpsplit.c's --sort, needed here so XOR-vs-previous-record is
   meaningful (same group, adjacent in the sorted stream). */
static u8 const * m4_sort_base;
static int cmp_rec_sorted( void const * a, void const * b ){
  u64 oa=*(u64 const*)a, ob=*(u64 const*)b;
  u8 const * pa=m4_sort_base+oa, * pb=m4_sort_base+ob;
  int c = memcmp(pa+64,pb+64,32); if(c) return c;
  u64 la=ld8(pa+24), lb=ld8(pb+24); if(la!=lb) return la<lb?-1:1;
  return memcmp(pa+32,pb+32,32);
}

/* ---------- literal stream + generic ZERO32/KEYREF/LITERAL walker ----- */

typedef struct { FILE * f; u64 n; } strm_t;
static strm_t sopen( char const * dir, char const * name ){
  char p[4096]; snprintf(p,sizeof(p),"%s/%s",dir,name);
  strm_t s; s.f=fopen(p,"wb"); s.n=0UL;
  if( !s.f ){ perror(p); exit(1); }
  return s;
}
static void sput( strm_t * s, void const * p, u64 n ){ fwrite(p,1,n,s->f); s->n+=n; }

/* walk `len` bytes at `d`, 32B-aligned ZERO/REPEAT/KEYREF windows else 4B
   literal steps; writes literal bytes to `lit`, tallies the byte counts and
   the decision entropy in `dec`, keyref fingerprints into kv/kn.

   `prev`, when non NULL, points at the same offset in the previous record of
   the same group, so a window identical to it costs one REPEAT symbol.  That
   is what "XOR against the previous record and zero-run code the result"
   buys -- without giving up the key dictionary for the windows that *did*
   change, which is the mistake that made the first M4 attempt worse than
   zstd. */
static void
generic_walk( u8 const * d, u8 const * prev, u64 len, int allow_keyref,
              strm_t * lit, u64 * zero_b, u64 * rep_b, u64 * keyref_b, u64 * lit_b,
              dec_t * dec, u64 ** kv, u64 * kn, u64 * kcap ){
  u8 zero32[32]; memset(zero32,0,32);
  u64 i=0UL;
  while( i+32UL<=len ){
    if( !ld8(d+i) && !memcmp(d+i,zero32,32) ){ *zero_b+=32UL; dec->zero++; i+=32UL; continue; }
    if( prev && !memcmp(d+i,prev+i,32) ){ *rep_b+=32UL; dec->repeat++; i+=32UL; continue; }
    if( allow_keyref ){
      u64 fp = key_fp(d+i);
      if( dict_has(fp) ){
        *keyref_b+=32UL; dec->keyref++;
        if( *kn==*kcap ){ *kcap = *kcap? (*kcap)*2UL : (1UL<<16); *kv=realloc(*kv,(*kcap)*8UL); }
        (*kv)[(*kn)++]=fp;
        i+=32UL; continue;
      }
    }
    sput(lit,d+i,4UL); *lit_b+=4UL; dec->miss++; i+=4UL;
  }
  if( len>i ){ sput(lit,d+i,len-i); *lit_b += len-i; }
}

/* ---------- b58 (for the group summary table) -------------------------- */
static void b58( u8 const * k, char * out ){
  static char const * A="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  u8 tmp[32]; memcpy(tmp,k,32); char buf[64]; int n=0;
  int zeros=0; while( zeros<32 && !tmp[zeros] ) zeros++;
  int start=zeros;
  while( start<32 ){
    int rem=0, nz=-1;
    for( int i=start;i<32;i++ ){ int cur=(rem<<8)|tmp[i]; tmp[i]=(u8)(cur/58); rem=cur%58; if(nz<0&&tmp[i]) nz=i; }
    buf[n++]=A[rem]; start = nz<0?32:nz;
  }
  int o=0; for(int i=0;i<zeros;i++) out[o++]='1'; while(n) out[o++]=buf[--n]; out[o]='\0';
}

/* ---------- known layouts (PLAN.md Phase 1 step 6) ---------------------

   The learned template of M3 only looks at 8 byte aligned slots and only
   calls one a pubkey when 90% of its values are in the dictionary, so it is
   blind by construction to the pubkey at offset 4 (SPL mint authority), 50
   (freeze authority), 76 (token delegate), 133 (close authority), 1 and 33
   (Metaplex update authority and mint) or 12/44/92/124 (stake), and it
   charges a wallet field whose keys are only 60% resident as free bytes.

   A layout declares the program's own struct instead: every field's offset
   and type, with no alignment or hit rate restriction.  The decoder
   dispatches on `owner` (plus data_len, plus Token-2022's account_type byte)
   rather than guessing per 32 byte window, so nothing here needs the
   dictionary to agree with a heuristic. */

static inline void kv_push( u64 ** kv, u64 * kn, u64 * kcap, u64 fp ){
  if( *kn==*kcap ){ *kcap = *kcap? (*kcap)*2UL : (1UL<<12); *kv=realloc(*kv,(*kcap)*8UL); }
  (*kv)[(*kn)++]=fp;
}

enum { LK_END=0, LK_PUBKEY, LK_NUM, LK_BLOB, LK_ZPAD, LK_ARR_PUBKEY, LK_TLV };

typedef struct { u16 off; u16 w; u8 kind; } lf_t;

static inline u64 ldn( u8 const * p, u64 w ){ u64 v=0UL; for( u64 i=0UL;i<w;i++ ) v |= (u64)p[i]<<(8UL*i); return v; }
static inline int nsig( u64 v ){ int n=0; while(v){ n++; v>>=8; } return n; }

/* spl_token::state::Account, 165 B */
static lf_t const L_tok_acct[] = {
  {   0, 32, LK_PUBKEY }, /* mint */
  {  32, 32, LK_PUBKEY }, /* owner (wallet) */
  {  64,  8, LK_NUM    }, /* amount */
  {  72,  4, LK_NUM    }, /* delegate COption tag */
  {  76, 32, LK_PUBKEY }, /* delegate */
  { 108,  1, LK_NUM    }, /* state */
  { 109,  4, LK_NUM    }, /* is_native COption tag */
  { 113,  8, LK_NUM    }, /* is_native (rent exempt reserve) */
  { 121,  8, LK_NUM    }, /* delegated_amount */
  { 129,  4, LK_NUM    }, /* close_authority COption tag */
  { 133, 32, LK_PUBKEY }, /* close_authority */
  {   0,  0, LK_END    }
};

/* spl_token::state::Mint, 82 B */
static lf_t const L_tok_mint[] = {
  {   0,  4, LK_NUM    }, /* mint_authority COption tag */
  {   4, 32, LK_PUBKEY }, /* mint_authority */
  {  36,  8, LK_NUM    }, /* supply */
  {  44,  1, LK_NUM    }, /* decimals */
  {  45,  1, LK_NUM    }, /* is_initialized */
  {  46,  4, LK_NUM    }, /* freeze_authority COption tag */
  {  50, 32, LK_PUBKEY }, /* freeze_authority */
  {   0,  0, LK_END    }
};

/* spl_token::state::Multisig, 355 B */
static lf_t const L_tok_multisig[] = {
  {   0,  1, LK_NUM       }, /* m */
  {   1,  1, LK_NUM       }, /* n */
  {   2,  1, LK_NUM       }, /* is_initialized */
  {   3,  0, LK_ARR_PUBKEY}, /* signers[11] */
  {   0,  0, LK_END       }
};

/* Token-2022 with extensions: the 165 B base, the account_type byte, then a
   TLV region.  Two variants, chosen by data[165] (1 = mint, 2 = account). */
static lf_t const L_t22_acct_ext[] = {
  {   0, 32, LK_PUBKEY }, {  32, 32, LK_PUBKEY }, {  64,  8, LK_NUM },
  {  72,  4, LK_NUM    }, {  76, 32, LK_PUBKEY }, { 108,  1, LK_NUM },
  { 109,  4, LK_NUM    }, { 113,  8, LK_NUM    }, { 121,  8, LK_NUM },
  { 129,  4, LK_NUM    }, { 133, 32, LK_PUBKEY },
  { 165,  1, LK_NUM    }, /* account_type */
  { 166,  0, LK_TLV    },
  {   0,  0, LK_END    }
};
static lf_t const L_t22_mint_ext[] = {
  {   0,  4, LK_NUM    }, {   4, 32, LK_PUBKEY }, {  36,  8, LK_NUM },
  {  44,  1, LK_NUM    }, {  45,  1, LK_NUM    }, {  46,  4, LK_NUM },
  {  50, 32, LK_PUBKEY },
  {  82, 83, LK_ZPAD   }, /* padded up to the 165 B account size */
  { 165,  1, LK_NUM    }, /* account_type */
  { 166,  0, LK_TLV    },
  {   0,  0, LK_END    }
};

/* solana stake account, 200 B (StakeStateV2::Stake) */
static lf_t const L_stake[] = {
  {   0,  4, LK_NUM    }, /* state discriminant */
  {   4,  8, LK_NUM    }, /* meta.rent_exempt_reserve */
  {  12, 32, LK_PUBKEY }, /* meta.authorized.staker */
  {  44, 32, LK_PUBKEY }, /* meta.authorized.withdrawer */
  {  76,  8, LK_NUM    }, /* meta.lockup.unix_timestamp */
  {  84,  8, LK_NUM    }, /* meta.lockup.epoch */
  {  92, 32, LK_PUBKEY }, /* meta.lockup.custodian */
  { 124, 32, LK_PUBKEY }, /* stake.delegation.voter_pubkey */
  { 156,  8, LK_NUM    }, /* stake.delegation.stake */
  { 164,  8, LK_NUM    }, /* activation_epoch */
  { 172,  8, LK_NUM    }, /* deactivation_epoch */
  { 180,  8, LK_NUM    }, /* warmup_cooldown_rate (f64) */
  { 188,  8, LK_NUM    }, /* credits_observed */
  { 196,  1, LK_NUM    }, /* stake_flags */
  { 197,  3, LK_ZPAD   },
  {   0,  0, LK_END    }
};

/* vote account, 3762 B -- only the fixed head is declared; the vote deque
   and the epoch credits behind it are left to the generic walker. */
static lf_t const L_vote[] = {
  {   0,  4, LK_NUM    }, /* version */
  {   4, 32, LK_PUBKEY }, /* node_pubkey */
  {  36, 32, LK_PUBKEY }, /* authorized_withdrawer */
  {  68,  1, LK_NUM    }, /* commission */
  {   0,  0, LK_END    }
};

/* address lookup table: a 56 B meta and then nothing but pubkeys */
static lf_t const L_alt[] = {
  {   0,  4, LK_NUM       }, /* type */
  {   4,  8, LK_NUM       }, /* deactivation_slot */
  {  12,  8, LK_NUM       }, /* last_extended_slot */
  {  20,  1, LK_NUM       }, /* last_extended_slot_start_index */
  {  21,  1, LK_NUM       }, /* authority COption tag */
  {  22, 32, LK_PUBKEY    }, /* authority */
  {  54,  2, LK_ZPAD      },
  {  56,  0, LK_ARR_PUBKEY}, /* addresses[] */
  {   0,  0, LK_END       }
};

/* durable nonce account (system program owned, 80 B) */
static lf_t const L_nonce[] = {
  {   0,  4, LK_NUM    }, /* version */
  {   4,  4, LK_NUM    }, /* state */
  {   8, 32, LK_PUBKEY }, /* authority */
  {  40, 32, LK_BLOB   }, /* durable nonce (a blockhash -- never in the dictionary) */
  {  72,  8, LK_NUM    }, /* fee_calculator.lamports_per_signature */
  {   0,  0, LK_END    }
};

/* Metaplex token metadata: the two pubkeys at misaligned offsets are the
   point; the borsh strings behind them are left to the generic walker. */
static lf_t const L_meta[] = {
  {   0,  1, LK_NUM    }, /* key (account discriminant) */
  {   1, 32, LK_PUBKEY }, /* update_authority */
  {  33, 32, LK_PUBKEY }, /* mint */
  {   0,  0, LK_END    }
};

/* Token-2022 extensions whose value holds pubkeys at known offsets. */
typedef struct { u16 type; u8 npk; u16 pk[2]; } ext_t;
static ext_t const exts[] = {
  {  1, 2, {  0, 32 } }, /* TransferFeeConfig: cfg authority, withdraw authority */
  {  3, 1, {  0,  0 } }, /* MintCloseAuthority */
  {  4, 1, {  0,  0 } }, /* ConfidentialTransferMint: authority */
  { 10, 1, {  0,  0 } }, /* InterestBearingConfig: rate authority */
  { 12, 1, {  0,  0 } }, /* PermanentDelegate */
  { 14, 2, {  0, 32 } }, /* TransferHook: authority, program_id */
  { 16, 1, {  0,  0 } }, /* ConfidentialTransferFeeConfig: authority */
  { 18, 2, {  0, 32 } }, /* MetadataPointer: authority, metadata_address */
  { 19, 2, {  0, 32 } }, /* TokenMetadata: update_authority, mint */
  { 20, 2, {  0, 32 } }, /* GroupPointer: authority, group_address */
  { 21, 2, {  0, 32 } }, /* TokenGroup: update_authority, mint */
  { 22, 2, {  0, 32 } }, /* GroupMemberPointer: authority, member_address */
  { 23, 2, {  0, 32 } }, /* TokenGroupMember: mint, group */
};
static ext_t const * ext_find( u32 t ){
  for( u64 i=0UL;i<sizeof(exts)/sizeof(exts[0]);i++ ) if( exts[i].type==t ) return exts+i;
  return NULL;
}

/* per field statistics.  A pubkey field carries its own {zero,keyref,escape}
   decision distribution and its own reference stream, which is the whole
   point of knowing the layout: the decision is per field, not per window. */
typedef struct {
  card_t card;                /* LK_NUM value alphabet */
  u64    nb[9], nbtot;        /* LK_NUM varint model: significant byte counts */
  u64    n;
  dec_t  dec;                 /* LK_PUBKEY/LK_BLOB/LK_ZPAD decisions */
  u64 *  kv; u64 kn, kcap;
} fstat_t;

/* where a model's byte/bit costs accumulate for one layout */
typedef struct {
  strm_t lit;
  u64    zero_b, keyref_b, lit_b, gen_b;
  dec_t  gdec;                /* generic region decisions */
  u64 *  kv;  u64 kn,  kcap;  /* references from declared fields */
  u64 *  gkv; u64 gkn, gkcap; /* references from generic regions */
} acc_t;

typedef struct {
  char const * name;
  char const * owner58;
  u8    owner[32];
  u64   lmin, lmax;
  int   t22;                  /* dispatch on data[165]: f_alt when it is 1 */
  lf_t const * f;     u64 nf;      fstat_t * fs;
  lf_t const * f_alt; u64 nf_alt;  fstat_t * fs_alt;
  u64   cnt, cnt_alt, raw;
  /* TLV region (Token-2022) */
  card_t tlv_type, tlv_len; u64 tlv_n, tlv_unknown;
  fstat_t tlv_pk;
  acc_t m5, m3;               /* same records priced under both models */
  double m3_u64_bits, m3_const_bytes;
  u64    m3_pub_hit, m3_pub_miss;
  double m5_u64_bits, m5_u64_bits_rob; /* learned template slots the layout did not claim */
  double m3_u64_bits_rob;
  u64    m5_pub_hit, m5_pub_miss;
} layout_t;

/* The program loaders: their accounts hold ELF images, which are the one part
   of the residual a schema cannot help with and an LZ stage can.  They get
   their own literal stream so that share is measurable. */
static char const * loader58[] = {
  "BPFLoaderUpgradeab1e11111111111111111111111",
  "BPFLoader2111111111111111111111111111111111",
  "BPFLoader1111111111111111111111111111111111",
  "LoaderV411111111111111111111111111111111111",
};
#define NLOADER (sizeof(loader58)/sizeof(loader58[0]))
static u8 loader_owner[NLOADER][32];
static int is_loader( u8 const * owner ){
  for( u64 i=0UL;i<NLOADER;i++ ) if( !memcmp(owner,loader_owner[i],32) ) return 1;
  return 0;
}

static layout_t layouts[] = {
  { .name="tok-acct",    .owner58="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", .lmin= 165, .lmax=165,     .f=L_tok_acct     },
  { .name="tok-mint",    .owner58="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", .lmin=  82, .lmax= 82,     .f=L_tok_mint     },
  { .name="tok-msig",    .owner58="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", .lmin= 355, .lmax=355,     .f=L_tok_multisig },
  { .name="t22-mint82",  .owner58="TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb", .lmin=  82, .lmax= 82,     .f=L_tok_mint     },
  { .name="t22-acct165", .owner58="TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb", .lmin= 165, .lmax=165,     .f=L_tok_acct     },
  { .name="t22-ext",     .owner58="TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb", .lmin= 166, .lmax=1UL<<20, .f=L_t22_acct_ext,
                         .f_alt=L_t22_mint_ext, .t22=1 },
  { .name="stake",       .owner58="Stake11111111111111111111111111111111111111", .lmin= 200, .lmax=200,     .f=L_stake        },
  { .name="vote",        .owner58="Vote111111111111111111111111111111111111111", .lmin=3762, .lmax=3762,    .f=L_vote         },
  { .name="alt",         .owner58="AddressLookupTab1e1111111111111111111111111", .lmin=  56, .lmax=1UL<<20, .f=L_alt          },
  { .name="nonce",       .owner58="11111111111111111111111111111111",            .lmin=  80, .lmax= 80,     .f=L_nonce        },
  { .name="meta",        .owner58="metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s", .lmin=  65, .lmax=1UL<<20, .f=L_meta         },
};
#define NLAYOUT (sizeof(layouts)/sizeof(layouts[0]))

static int
b58_dec( char const * s, u8 * out ){
  static char const * A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  u8 acc[32]; memset(acc,0,32);
  for( char const * p=s; *p; p++ ){
    char const * q = strchr(A,*p); if(!q) return -1;
    int carry = (int)(q-A);
    for( int i=31;i>=0;i-- ){ int t=acc[i]*58+carry; acc[i]=(u8)t; carry=t>>8; }
    if( carry ) return -1;
  }
  memcpy(out,acc,32); return 0;
}

static u64 lf_count( lf_t const * f ){ u64 n=0UL; while( f[n].kind!=LK_END ) n++; return n; }
static fstat_t * fs_new( u64 n ){
  fstat_t * fs = calloc(n,sizeof(fstat_t));
  for( u64 i=0UL;i<n;i++ ) card_init(&fs[i].card);
  return fs;
}

static void
layouts_init( char const * outdir ){
  char p[4096];
  snprintf(p,sizeof(p),"%s/m5",outdir);   mkdir(p,0755);
  snprintf(p,sizeof(p),"%s/m3lay",outdir); mkdir(p,0755);
  for( u64 i=0UL;i<NLOADER;i++ )
    if( b58_dec(loader58[i],loader_owner[i]) ){ fprintf(stderr,"bad b58 %s\n",loader58[i]); exit(1); }
  for( u64 i=0UL;i<NLAYOUT;i++ ){
    layout_t * L = layouts+i;
    if( b58_dec(L->owner58,L->owner) ){ fprintf(stderr,"bad b58 %s\n",L->owner58); exit(1); }
    L->nf = lf_count(L->f); L->fs = fs_new(L->nf);
    if( L->f_alt ){ L->nf_alt = lf_count(L->f_alt); L->fs_alt = fs_new(L->nf_alt); }
    card_init(&L->tlv_type); card_init(&L->tlv_len); card_init(&L->tlv_pk.card);
    char n[256];
    snprintf(n,sizeof(n),"m5/%s.bin",L->name);    L->m5.lit = sopen(outdir,n);
    snprintf(n,sizeof(n),"m3lay/%s.bin",L->name); L->m3.lit = sopen(outdir,n);
  }
}

/* owner + data_len -> layout.  Token-2022's mint/account variants share one
   entry and are separated inside m5_record by the account_type byte, so that
   every (owner,data_len) group maps to exactly one layout. */
static layout_t *
pick_layout( u8 const * owner, u64 len ){
  for( u64 i=0UL;i<NLAYOUT;i++ ){
    layout_t * L = layouts+i;
    if( len<L->lmin || len>L->lmax ) continue;
    if( memcmp(owner,L->owner,32) ) continue;
    return L;
  }
  return NULL;
}

static void
price_pubkey( fstat_t * fs, u8 const * v, acc_t * a ){
  static u8 const z32[32]={0};
  fs->n++;
  if( !memcmp(v,z32,32) ){ fs->dec.zero++; a->zero_b+=32UL; return; }
  u64 fp=key_fp(v);
  if( dict_has(fp) ){
    fs->dec.keyref++; a->keyref_b+=32UL;
    kv_push(&a->kv,&a->kn,&a->kcap,fp);
    kv_push(&fs->kv,&fs->kn,&fs->kcap,fp);
  } else { fs->dec.miss++; sput(&a->lit,v,32UL); a->lit_b+=32UL; }
}

static void
price_num( fstat_t * fs, u64 v ){
  fs->n++; card_add(&fs->card,v);
  int b=nsig(v); fs->nb[b]++; fs->nbtot += (u64)b;
}

static void
gen_region( acc_t * a, u8 const * d, u64 len ){
  u64 rep=0UL;
  a->gen_b += len;
  generic_walk(d,NULL,len,1,&a->lit,&a->zero_b,&rep,&a->keyref_b,&a->lit_b,&a->gdec,&a->gkv,&a->gkn,&a->gkcap);
}

/* entropy of two reference streams coded as one, which is what M2/M3 do and
   what the M5 headline charges per layout */
static double
kref_entropy2( u64 const * a, u64 na, u64 const * b, u64 nb ){
  if( !na && !nb ) return 0.0;
  u64 * t=malloc((na+nb)*8UL);
  if( na ) memcpy(t,a,na*8UL);
  if( nb ) memcpy(t+na,b,nb*8UL);
  double h=kref_entropy_bits(t,na+nb);
  free(t);
  return h;
}

/* total bits of a two symbol decision (template pubkey hit vs escape) */
static double
bin_bits( u64 a, u64 b ){
  u64 t=a+b; if(!t) return 0.0;
  double h=0.0;
  if( a ) h -= (double)a*log2((double)a/(double)t);
  if( b ) h -= (double)b*log2((double)b/(double)t);
  return h;
}

/* M3's own record pricing, factored out so that the global pass and the
   per layout pass (step 6's comparison) cannot drift apart. */
static void
m3_record( grp_t * g, u8 const * data, u64 dlen, acc_t * a, u64 * pub_hit, u64 * pub_miss ){
  if( g && g->templated ){
    for( u64 j=0UL;j<g->nslots; ){
      slot_t * s=g->field+j;
      if( s->ftype==FT_PUBKEY ){
        u8 const * v=data+j*FLD8;
        u64 fp=key_fp(v);
        if( dict_has(fp) ){
          (*pub_hit)++; a->keyref_b+=32UL; a->gdec.keyref++;
          kv_push(&a->kv,&a->kn,&a->kcap,fp);
        } else {
          (*pub_miss)++; sput(&a->lit,v,32UL); a->lit_b+=32UL; a->gdec.miss++;
        }
        j+=4UL;
      } else j++;  /* FT_CONST/FT_U64 priced from aggregates, FT_FREE via spans */
    }
    for( u64 si=0UL; si<g->nspans; si++ ) gen_region(a,data+g->spans[si].off,g->spans[si].len);
  } else gen_region(a,data,dlen);
}

/* A byte range the layout does not declare.  It is not free bytes: the
   learned template of M3 still knows this group, so the range is priced
   through its const/u64/pubkey slots and only what is left over goes to the
   generic walker.  Composing the two is what keeps M5 >= M3 everywhere. */
static void
tmpl_region( grp_t * g, acc_t * a, u8 const * d, u64 lo, u64 hi,
             u64 * pub_hit, u64 * pub_miss ){
  if( hi<=lo ) return;
  if( !g || !g->templated ){ gen_region(a,d+lo,hi-lo); return; }
  u64 first=(lo+FLD8-1UL)/FLD8, last=hi/FLD8;
  if( last>g->nslots ) last=g->nslots;
  if( first>=last ){ gen_region(a,d+lo,hi-lo); return; }
  if( first*FLD8>lo ) gen_region(a,d+lo,first*FLD8-lo);
  u64 j=first;
  while( j<last ){
    slot_t * s=g->field+j;
    if( s->ftype==FT_CONST || s->ftype==FT_U64 ){ s->m5_used++; j++; continue; }
    if( s->ftype==FT_PUBKEY && j+4UL<=last ){
      s->m5_used++;
      u8 const * v=d+j*FLD8; u64 fp=key_fp(v);
      if( dict_has(fp) ){
        (*pub_hit)++; a->keyref_b+=32UL; a->gdec.keyref++;
        kv_push(&a->kv,&a->kn,&a->kcap,fp);
      } else { (*pub_miss)++; sput(&a->lit,v,32UL); a->lit_b+=32UL; a->gdec.miss++; }
      j+=4UL; continue;
    }
    u64 k=j;                                    /* run of slots the template cannot help with */
    while( k<last ){
      u8 t=g->field[k].ftype;
      if( t==FT_CONST || t==FT_U64 || (t==FT_PUBKEY && k+4UL<=last) ) break;
      k++;
    }
    gen_region(a,d+j*FLD8,(k-j)*FLD8);
    j=k;
  }
  if( last*FLD8<hi ) gen_region(a,d+last*FLD8,hi-last*FLD8);
}

/* Token-2022 TLV: (type u16, len u16, value) triples.  The headers are cheap
   categorical symbols; a known extension's pubkeys are declared, anything
   else falls back to the generic walker. */
static u64
price_tlv( layout_t * L, grp_t * g, u8 const * d, u64 off, u64 len, acc_t * a ){
  u64 p=off;
  while( p+4UL<=len ){
    u32 t  = (u32)d[p]   | ((u32)d[p+1]<<8);
    u32 vl = (u32)d[p+2] | ((u32)d[p+3]<<8);
    if( !t ) break;                       /* uninitialized tail */
    if( p+4UL+(u64)vl > len ) break;      /* malformed: let the generic walker have it */
    L->tlv_n++; card_add(&L->tlv_type,t); card_add(&L->tlv_len,vl);
    u64 vo = p+4UL;                   /* value offset inside the record */
    u8 const * v = d+vo;
    ext_t const * e = ext_find(t);
    if( !e ) L->tlv_unknown++;
    u64 vp=0UL;
    if( e ) for( int k=0;k<e->npk;k++ ){
      u64 o=e->pk[k];
      if( o+32UL<=(u64)vl ){
        if( o>vp ) tmpl_region(g,a,d,vo+vp,vo+o,&L->m5_pub_hit,&L->m5_pub_miss);
        price_pubkey(&L->tlv_pk,v+o,a);
        vp=o+32UL;
      }
    }
    if( vp<(u64)vl ) tmpl_region(g,a,d,vo+vp,vo+(u64)vl,&L->m5_pub_hit,&L->m5_pub_miss);
    p += 4UL+(u64)vl;
  }
  return p;
}

/* price one record under its layout (M5) */
static void
m5_record( layout_t * L, grp_t * g, u8 const * d, u64 len, acc_t * a ){
  lf_t const * f = L->f; u64 nf=L->nf; fstat_t * fs=L->fs;
  if( L->t22 && len>165UL && d[165]==1 ){ f=L->f_alt; nf=L->nf_alt; fs=L->fs_alt; L->cnt_alt++; }
  else L->cnt++;
  L->raw += len;
  u64 pos=0UL;
  for( u64 i=0UL;i<nf;i++ ){
    u64 off=f[i].off, w=f[i].w;
    if( off>=len ) break;
    if( off>pos ) tmpl_region(g,a,d,pos,off,&L->m5_pub_hit,&L->m5_pub_miss);
    else if( off<pos ) continue;          /* shouldn't happen: fields are ordered */
    switch( f[i].kind ){
      case LK_PUBKEY:
        if( off+32UL<=len ){ price_pubkey(fs+i,d+off,a); pos=off+32UL; }
        break;
      case LK_NUM:
        if( off+w<=len ){ price_num(fs+i,ldn(d+off,w)); pos=off+w; }
        break;
      case LK_BLOB:
        if( off+w<=len ){ fs[i].n++; sput(&a->lit,d+off,w); a->lit_b+=w; pos=off+w; }
        break;
      case LK_ZPAD: {
        u64 n = off+w<=len ? w : len-off;
        fs[i].n++;
        int z=1; for( u64 k=0UL;k<n;k++ ) if( d[off+k] ){ z=0; break; }
        if( z ) { fs[i].dec.zero++; a->zero_b+=n; }
        else { fs[i].dec.miss++; sput(&a->lit,d+off,n); a->lit_b+=n; }
        pos=off+n;
        break;
      }
      case LK_ARR_PUBKEY: {
        u64 q=off;
        while( q+32UL<=len ){ price_pubkey(fs+i,d+q,a); q+=32UL; }
        pos=q;
        break;
      }
      case LK_TLV:
        pos = price_tlv(L,g,d,off,len,a);
        break;
      default: break;
    }
  }
  if( pos<len ) tmpl_region(g,a,d,pos,len,&L->m5_pub_hit,&L->m5_pub_miss);
}

/* cost of a LK_NUM field: the cheaper of an order-0 code over its value
   alphabet and a varint (ANS coded significant byte count + raw payload).
   Amount-like fields blow past the 64k alphabet cap, where M3 would have
   dumped them into the literal stream instead. */
static double
fs_varint_bits( fstat_t const * f ){
  double hn=0.0;
  if( f->n ) for( int i=0;i<9;i++ ) if( f->nb[i] ){
    double p=(double)f->nb[i]/(double)f->n; hn -= (double)f->nb[i]*log2(p);
  }
  return hn + 8.0*(double)f->nbtot;
}

static double
fs_num_bits( fstat_t const * f, int * varint ){
  double hc = card_entropy(&f->card);
  double raw = hc>=0.0 ? hc*(double)f->n : 1e30;
  double var = fs_varint_bits(f);
  if( varint ) *varint = raw<=var ? 0 : 1;
  return raw<var ? raw : var;
}

/* An order-0 alphabet measured on a 1/128 sample cannot cost more than
   log2(records seen) bits, so a field that is near unique per record here is
   priced by the sample size rather than by the field.  This variant charges
   such a field as a varint instead, which is sample independent, and is the
   honest upper end of the same measurement. */
static int
fs_near_unique( fstat_t const * f ){
  return !f->card.over && f->n && (double)f->card.cnt >= 0.5*(double)f->n;
}
static double
fs_num_bits_robust( fstat_t const * f ){
  if( fs_near_unique(f) ) return fs_varint_bits(f);
  return fs_num_bits(f,NULL);
}

static double
layout_field_bits( lf_t const * f, u64 nf, fstat_t const * fs, int robust ){
  double bits=0.0;
  for( u64 i=0UL;i<nf;i++ ){
    switch( f[i].kind ){
      case LK_NUM:    bits += robust ? fs_num_bits_robust(fs+i) : fs_num_bits(fs+i,NULL); break;
      case LK_PUBKEY:
      case LK_ARR_PUBKEY:
      case LK_ZPAD:   bits += dec_bits(&fs[i].dec);   break;
      default: break;
    }
  }
  return bits;
}

int
main( int argc, char ** argv ){
  if( argc<4 ){ fprintf(stderr,"usage: %s keys.bin corpus outdir [topN] [maxlen]\n",argv[0]); return 1; }
  u64 topn   = argc>4 ? strtoull(argv[4],NULL,10) : 1000UL;
  u64 maxlen = argc>5 ? strtoull(argv[5],NULL,10) : 16384UL;
  char * outdir = argv[3];
  mkdir(outdir,0755);
  layouts_init(outdir);

  /* load the dictionary: every account pubkey in the snapshot */
  {
    int kfd=open(argv[1],O_RDONLY); if(kfd<0){ perror("open keys"); return 1; }
    struct stat ks; fstat(kfd,&ks); u64 nkeys=(u64)ks.st_size/32UL;
    u64 lg=20UL; while( (1UL<<lg) < nkeys*2UL ) lg++;
    fprintf(stderr,"dictionary: %lu keys, table 2^%lu (%.1f GiB)\n",nkeys,lg,(double)(1UL<<lg)*8.0/1073741824.0);
    dict_init(lg);
    u64 bufsz=32UL<<20; u8 * buf=malloc(bufsz); u64 done=0UL;
    for(;;){
      ssize_t n=read(kfd,buf,bufsz);
      if( n<0 ){ perror("read keys"); return 1; }
      if( !n ) break;
      for( ssize_t i=0;i+32<=n;i+=32 ) dict_add(key_fp(buf+i));
      done += (u64)n/32UL;
      if( !(done%(1UL<<26)) ) fprintf(stderr,"\r  loaded %lu keys",done), fflush(stderr);
    }
    free(buf); close(kfd);
    fprintf(stderr,"\r  loaded %lu keys\n",done);
  }

  int fd=open(argv[2],O_RDONLY); if(fd<0){ perror("open corpus"); return 1; }
  struct stat st; fstat(fd,&st); fsz=(u64)st.st_size;
  u8 * m=mmap(NULL,fsz,PROT_READ,MAP_PRIVATE,fd,0);
  if( m==MAP_FAILED ){ perror("mmap corpus"); return 1; }
  madvise(m,fsz,MADV_SEQUENTIAL);
  if( memcmp(m,"SNAPCORP1",9) ){ fprintf(stderr,"bad magic\n"); return 1; }
  base=m;

  gt_init(20UL);

  /* ---- pass 1: metadata contexts + group population (cheap, whole corpus) */
  u64 n=0UL, data_bytes=0UL;
  {
    rec_t r; u64 off=16UL;
    while( next_rec(&off,&r) ){
      n++; data_bytes += r.data_len;
      grp_t * g = gt_get(r.owner,r.data_len);
      if( !g->cnt ){ card_init(&g->lam); card_init(&g->rent); card_init(&g->exe); }
      g->cnt++;
      card_add(&g->lam,r.lamports); card_add(&g->rent,r.rent_epoch); card_add(&g->exe,(u64)r.exec);
      grp_t * go = gt_get(r.owner,(u64)-1);
      if( !go->cnt ) card_init(&go->dlen_of_owner);
      go->cnt++; card_add(&go->dlen_of_owner,r.data_len);
    }
  }
  fprintf(stderr,"pass 1: %lu records, %lu groups\n",n,gt_cnt);

  /* which groups a hand written layout covers (M5) */
  { u64 lay_grp=0UL, lay_rec=0UL;
    for( u64 i=0UL;i<gt_cap;i++ ) if( gt[i].used && gt[i].len!=(u64)-1 ){
      gt[i].lay = pick_layout(gt[i].owner,gt[i].len);
      if( gt[i].lay ){ lay_grp++; lay_rec += gt[i].cnt; }
    }
    fprintf(stderr,"layouts: %lu groups, %lu records (%.1f%%) covered by a known layout\n",
            lay_grp,lay_rec,100.0*(double)lay_rec/(double)(n?n:1));
  }

  /* ---- select template groups: top-N by count, len in (0,maxlen] -------- */
  grp_t ** cand = malloc(gt_cnt*sizeof(grp_t*)); u64 ncand=0UL;
  for( u64 i=0UL;i<gt_cap;i++ ) if( gt[i].used && gt[i].len!=(u64)-1 && gt[i].len && gt[i].len<=maxlen )
    cand[ncand++]=gt+i;
  qsort(cand,ncand,sizeof(grp_t*),cmp_grp_cnt);
  u64 ntemplated = ncand<topn ? ncand : topn;
  u64 templated_records=0UL;
  for( u64 i=0UL;i<ntemplated;i++ ){
    grp_t * g = cand[i];
    g->templated = 1;
    g->nslots = g->len/FLD8;
    g->field = calloc(g->nslots,sizeof(slot_t));
    for( u64 j=0UL;j<g->nslots;j++ ) card_init(&g->field[j].card);
    templated_records += g->cnt;
  }
  fprintf(stderr,"templates: %lu/%lu candidate groups selected, covering %lu/%lu records (%.1f%%)\n",
          ntemplated,ncand,templated_records,n,100.0*(double)templated_records/(double)(n?n:1));

  /* ---- pass 2: per-slot const / pubkey-hit-rate / cardinality, templated groups only */
  {
    rec_t r; u64 off=16UL;
    while( next_rec(&off,&r) ){
      if( !r.data_len || r.data_len>maxlen ) continue;
      grp_t * g = gt_get(r.owner,r.data_len);
      if( !g->templated ) continue;
      int first = !g->tmpl_seen; g->tmpl_seen=1;
      for( u64 j=0UL;j<g->nslots;j++ ){
        slot_t * s = g->field+j;
        u8 const * v = r.data+j*FLD8;
        if( first ){ s->const_ok=1; memcpy(s->const_val,v,8UL); }
        else if( s->const_ok && memcmp(s->const_val,v,8UL) ) s->const_ok=0;
        u64 sv=ld8(v);
        card_add(&s->card,sv);
        int nsb=nsig(sv); s->nb[nsb]++; s->nbtot += (u64)nsb;
        if( (j&3UL)==0UL && j+4UL<=g->nslots ){ /* 32B window start, fully inside the templated region */
          s->tot++;
          if( dict_has(key_fp(v)) ) s->hit++;
        }
      }
    }
  }

  /* ---- finalize per-group templates -------------------------------------- */
  for( u64 gi=0UL; gi<ntemplated; gi++ ){
    grp_t * g = cand[gi];
    for( u64 j=0UL;j<g->nslots; ){
      if( (j&3UL)==0UL && j+4UL<=g->nslots ){
        int all_const=1; for( u64 k=0;k<4UL;k++ ) if( !g->field[j+k].const_ok ) { all_const=0; break; }
        if( all_const ){ for(u64 k=0;k<4UL;k++) g->field[j+k].ftype=FT_CONST; j+=4UL; continue; }
        double rate = g->field[j].tot ? (double)g->field[j].hit/(double)g->field[j].tot : 0.0;
        if( rate>=PUBKEY_THRESH ){
          g->field[j].ftype=FT_PUBKEY;
          for(u64 k=1;k<4UL;k++) g->field[j+k].ftype=FT_PUBKEY_CONT;
          j+=4UL; continue;
        }
      }
      slot_t * s=g->field+j;
      if( s->const_ok ) s->ftype=FT_CONST;
      else if( card_entropy(&s->card)>=0.0 ) s->ftype=FT_U64;
      else s->ftype=FT_FREE;
      j++;
    }
    /* merge contiguous FT_FREE slots (+trailing tail bytes) into spans */
    u64 tail = g->len - g->nslots*FLD8;
    span_t * sp = malloc((g->nslots+1UL)*sizeof(span_t)); u64 ns=0UL;
    u64 j=0UL;
    while( j<g->nslots ){
      if( g->field[j].ftype!=FT_FREE ){ j++; continue; }
      u64 k=j; while( k<g->nslots && g->field[k].ftype==FT_FREE ) k++;
      sp[ns].off=j*FLD8; sp[ns].len=(k-j)*FLD8; ns++;
      j=k;
    }
    if( tail ){
      if( ns && sp[ns-1].off+sp[ns-1].len==g->nslots*FLD8 ) sp[ns-1].len+=tail;
      else { sp[ns].off=g->nslots*FLD8; sp[ns].len=tail; ns++; }
    }
    g->spans=sp; g->nspans=ns;
    for( u64 i=0UL;i<g->nslots;i++ ){
      if( g->field[i].ftype==FT_CONST ) g->const_bytes += FLD8*g->cnt;
      else if( g->field[i].ftype==FT_U64 ) g->u64_bytes += FLD8*g->cnt;
      else if( g->field[i].ftype==FT_PUBKEY ) g->pubkey_bytes += 32UL*g->cnt;
    }
  }

  /* ---- M1: metadata cost (H(owner), H(len|owner), H(lam|owner,len), ...) */
  double h_owner=0.0, h_dlen=0.0, h_lam=0.0, h_rent=0.0, h_exe=0.0;
  u64 owner_tot=0UL;
  for( u64 i=0UL;i<gt_cap;i++ ) if( gt[i].used && gt[i].len==(u64)-1 ) owner_tot+=gt[i].cnt;
  for( u64 i=0UL;i<gt_cap;i++ ){
    grp_t * g=gt+i; if( !g->used ) continue;
    if( g->len==(u64)-1 ){
      double p=(double)g->cnt/(double)owner_tot; h_owner -= p*log2(p);
      double h=card_entropy(&g->dlen_of_owner); if(h>=0.0) h_dlen += p*h;
      continue;
    }
    double w=(double)g->cnt/(double)n;
    double a=card_entropy(&g->lam), b=card_entropy(&g->rent), c=card_entropy(&g->exe);
    h_lam  += w*(a>=0.0?a:16.0); h_rent += w*(b>=0.0?b:2.0); h_exe += w*(c>=0.0?c:1.0);
  }
  double pubkey_bits = 256.0 - log2((double)(n?n:1)) + 1.44;
  double meta_bits = h_owner+h_dlen+h_lam+h_rent+h_exe;

  /* ---- pass 3: M2 (no template) + M3 (template) tokenization ------------- */
  acc_t m2acc, m3acc;
  memset(&m2acc,0,sizeof(m2acc)); memset(&m3acc,0,sizeof(m3acc));
  m2acc.lit = sopen(outdir,"m2_literal.bin");
  m3acc.lit = sopen(outdir,"m3_literal.bin");
  u64 pub_hit=0UL, pub_miss=0UL;
  {
    rec_t r; u64 off=16UL;
    while( next_rec(&off,&r) ){
      if( !r.data_len ) continue;
      gen_region(&m2acc,r.data,r.data_len);
      grp_t * g = r.data_len<=maxlen ? gt_get(r.owner,r.data_len) : NULL;
      m3_record(g,r.data,r.data_len,&m3acc,&pub_hit,&pub_miss);
    }
  }
  fclose(m2acc.lit.f); fclose(m3acc.lit.f);
  u64 m2_zero=m2acc.zero_b, m2_keyref_b=m2acc.keyref_b, m2_lit_b=m2acc.lit_b, m2kn=m2acc.gkn;
  u64 m3_zero=m3acc.zero_b, m3_keyref_b=m3acc.keyref_b, m3_lit_b=m3acc.lit_b;
  u64 m3kn=m3acc.kn+m3acc.gkn;
  dec_t m2_dec=m2acc.gdec, m3_dec=m3acc.gdec;

  double m2_keyref_bits = kref_entropy_bits(m2acc.gkv,m2acc.gkn);
  double m3_keyref_bits = kref_entropy2(m3acc.kv,m3acc.kn,m3acc.gkv,m3acc.gkn);
  double m2_dec_bits = dec_bits(&m2_dec);
  double m3_dec_bits = dec_bits(&m3_dec);
  double m3_u64_bits=0.0;
  for( u64 gi=0UL; gi<ntemplated; gi++ ){
    grp_t * g=cand[gi];
    for( u64 j=0UL;j<g->nslots;j++ ) if( g->field[j].ftype==FT_U64 ){
      double h=card_entropy(&g->field[j].card);
      m3_u64_bits += g->cnt * (h>=0.0?h:64.0);
    }
  }
  double pub_bit = (pub_hit+pub_miss) ? -( (double)pub_hit*log2((double)pub_hit/(double)(pub_hit+pub_miss))
                                          +(double)pub_miss*log2((double)pub_miss/(double)(pub_hit+pub_miss)) ) : 0.0;

  /* ---- pass 4: sorted (owner,len,pubkey) order, two variants -------------
     M4a: XOR the whole record against the previous same-group record and
          tokenize the result with no dictionary -- the literal reading of
          "XOR and zero-run code", kept because it is a useful negative.
     M4b: keep M3's template and the key dictionary, and add a REPEAT token
          for any 32B window (or pubkey field) equal to the previous record's
          window at the same offset.  This is what IDEAS.md §5 actually wants
          ("constant field" and "same mint as the last one"), and it never
          gives up a KEYREF for a field that legitimately changed. */
  u64 cap=1UL<<20, cnt=0UL; u64 * idx=malloc(cap*8UL);
  { rec_t r; u64 off=16UL; while(next_rec(&off,&r)){ if(cnt==cap){cap*=2UL; idx=realloc(idx,cap*8UL);} idx[cnt++]=off-104UL-r.data_len; } }
  m4_sort_base = base;
  qsort(idx,cnt,8UL,cmp_rec_sorted);

  strm_t m4_lit = sopen(outdir,"m4_literal.bin");
  u64 m4_zero=0UL,m4_rep_b=0UL,m4_lit_b=0UL; dec_t m4_dec={0,0,0,0};
  {
    u64 scratch_cap = 2UL<<20; u8 * scratch=malloc(scratch_cap);
    u8 const * prev=NULL; u64 prev_len=0UL; u8 prev_owner[32]; int have_prev=0;
    for( u64 i=0UL;i<cnt;i++ ){
      u8 const * p = base+idx[i];
      u64 dlen = ld8(p+24); u8 const * owner=p+64; u8 const * data=p+104;
      int same_group = have_prev && dlen==prev_len && !memcmp(owner,prev_owner,32);
      if( !dlen ){ have_prev=0; continue; }
      if( !same_group ){
        generic_walk(data,NULL,dlen,0,&m4_lit,&m4_zero,&m4_rep_b,NULL,&m4_lit_b,&m4_dec,NULL,NULL,NULL);
      } else {
        if( dlen>scratch_cap ){ scratch_cap=dlen; scratch=realloc(scratch,scratch_cap); }
        for( u64 k=0UL;k<dlen;k++ ) scratch[k]=data[k]^prev[k];
        generic_walk(scratch,NULL,dlen,0,&m4_lit,&m4_zero,&m4_rep_b,NULL,&m4_lit_b,&m4_dec,NULL,NULL,NULL);
      }
      prev=data; prev_len=dlen; memcpy(prev_owner,owner,32); have_prev=1;
    }
    free(scratch);
  }
  fclose(m4_lit.f);
  double m4_dec_bits = dec_bits(&m4_dec); /* only {zero,miss}, keyref unused */

  /* pass 4b stats: XOR delta alphabet of each templated u64 slot, in sorted
     order (the raw alphabet in `card` is what M3 charges; the delta alphabet
     is what M4b charges for every record that has a predecessor). */
  for( u64 gi=0UL; gi<ntemplated; gi++ ){
    grp_t * g=cand[gi];
    for( u64 j=0UL;j<g->nslots;j++ ) card_init(&g->field[j].cardD);
  }
  {
    u8 const * prev=NULL; u64 prev_len=0UL; u8 prev_owner[32]; int have_prev=0;
    for( u64 i=0UL;i<cnt;i++ ){
      u8 const * p = base+idx[i];
      u64 dlen = ld8(p+24); u8 const * owner=p+64; u8 const * data=p+104;
      int same_group = have_prev && dlen==prev_len && !memcmp(owner,prev_owner,32);
      if( !dlen ){ have_prev=0; continue; }
      grp_t * g = dlen<=maxlen ? gt_get(owner,dlen) : NULL;
      if( same_group && g && g->templated )
        for( u64 j=0UL;j<g->nslots;j++ ) if( g->field[j].ftype==FT_U64 )
          card_add(&g->field[j].cardD, ld8(data+j*FLD8)^ld8(prev+j*FLD8));
      prev=data; prev_len=dlen; memcpy(prev_owner,owner,32); have_prev=1;
    }
  }

  strm_t m4b_lit = sopen(outdir,"m4b_literal.bin");
  u64 m4b_zero=0UL,m4b_rep_b=0UL,m4b_keyref_b=0UL,m4b_lit_b=0UL; dec_t m4b_dec={0,0,0,0};
  u64 *m4bkv=NULL,m4bkn=0UL,m4bkcap=0UL;
  u64 m4b_pubrep=0UL, m4b_pubhit=0UL, m4b_pubmiss=0UL;
  {
    u8 const * prev=NULL; u64 prev_len=0UL; u8 prev_owner[32]; int have_prev=0;
    for( u64 i=0UL;i<cnt;i++ ){
      u8 const * p = base+idx[i];
      u64 dlen = ld8(p+24); u8 const * owner=p+64; u8 const * data=p+104;
      int same_group = have_prev && dlen==prev_len && !memcmp(owner,prev_owner,32);
      if( !dlen ){ have_prev=0; continue; }
      u8 const * pv = same_group ? prev : NULL;
      grp_t * g = dlen<=maxlen ? gt_get(owner,dlen) : NULL;
      if( g && g->templated ){
        for( u64 j=0UL;j<g->nslots; ){
          slot_t * s=g->field+j;
          if( s->ftype==FT_PUBKEY ){
            u8 const * v=data+j*FLD8;
            if( pv && !memcmp(v,pv+j*FLD8,32) ){ m4b_pubrep++; m4b_rep_b+=32UL; m4b_dec.repeat++; }
            else {
              u64 fp=key_fp(v);
              if( dict_has(fp) ){
                m4b_pubhit++; m4b_keyref_b+=32UL; m4b_dec.keyref++;
                if(m4bkn==m4bkcap){m4bkcap=m4bkcap?m4bkcap*2UL:(1UL<<16); m4bkv=realloc(m4bkv,m4bkcap*8UL);}
                m4bkv[m4bkn++]=fp;
              } else { m4b_pubmiss++; sput(&m4b_lit,v,32UL); m4b_lit_b+=32UL; m4b_dec.miss++; }
            }
            j+=4UL;
          } else j++;
        }
        for( u64 si=0UL; si<g->nspans; si++ )
          generic_walk(data+g->spans[si].off, pv?pv+g->spans[si].off:NULL, g->spans[si].len, 1,
                       &m4b_lit,&m4b_zero,&m4b_rep_b,&m4b_keyref_b,&m4b_lit_b,&m4b_dec,&m4bkv,&m4bkn,&m4bkcap);
      } else {
        generic_walk(data,pv,dlen,1,&m4b_lit,&m4b_zero,&m4b_rep_b,&m4b_keyref_b,&m4b_lit_b,&m4b_dec,&m4bkv,&m4bkn,&m4bkcap);
      }
      prev=data; prev_len=dlen; memcpy(prev_owner,owner,32); have_prev=1;
    }
  }
  fclose(m4b_lit.f);
  free(idx);
  double m4b_keyref_bits = kref_entropy_bits(m4bkv,m4bkn);
  double m4b_dec_bits = dec_bits(&m4b_dec);
  /* u64 template fields: delta vs raw is a per field choice the encoder makes
     once in the preamble, so charge the cheaper of the two.  Reported both
     ways because the delta alphabet losing is itself the interesting result --
     sorting by pubkey leaves adjacent records uncorrelated in these fields,
     so XOR just doubles the alphabet. */
  double m4b_u64_bits=0.0, m4b_u64_delta_only=0.0; u64 m4b_u64_delta_wins=0UL, m4b_u64_slots=0UL;
  for( u64 gi=0UL; gi<ntemplated; gi++ ){
    grp_t * g=cand[gi];
    for( u64 j=0UL;j<g->nslots;j++ ) if( g->field[j].ftype==FT_U64 ){
      double hr=card_entropy(&g->field[j].card), hd=card_entropy(&g->field[j].cardD);
      if( hr<0.0 ) hr=64.0;
      if( hd<0.0 ) hd=64.0;
      m4b_u64_slots++;
      if( hd<hr ) m4b_u64_delta_wins++;
      m4b_u64_delta_only += hr + (double)(g->cnt-1UL)*hd;
      m4b_u64_bits       += hr + (double)(g->cnt-1UL)*(hd<hr?hd:hr);
    }
  }

  /* ---- pass 3b: M3 again, restricted to the records a layout covers, so
     step 6 can be priced against M3 on exactly the same bytes ------------- */
  {
    rec_t r; u64 off=16UL;
    while( next_rec(&off,&r) ){
      if( !r.data_len ) continue;
      layout_t * L = pick_layout(r.owner,r.data_len);
      if( !L ) continue;
      grp_t * g = r.data_len<=maxlen ? gt_get(r.owner,r.data_len) : NULL;
      m3_record(g,r.data,r.data_len,&L->m3,&L->m3_pub_hit,&L->m3_pub_miss);
    }
  }
  for( u64 gi=0UL; gi<ntemplated; gi++ ){
    grp_t * g=cand[gi]; layout_t * L=(layout_t*)g->lay;
    if( !L ) continue;
    for( u64 j=0UL;j<g->nslots;j++ ) if( g->field[j].ftype==FT_U64 ){
      slot_t * sl=g->field+j;
      double h=card_entropy(&sl->card);
      double cb=(double)g->cnt*(h>=0.0?h:64.0);
      double vb=slot_varint_bits(sl,g->cnt);
      int nu = !sl->card.over && (double)sl->card.cnt >= 0.5*(double)g->cnt;
      L->m3_u64_bits     += cb;
      L->m3_u64_bits_rob += nu ? vb : cb;
    }
    L->m3_const_bytes += (double)g->const_bytes;
  }

  /* ---- pass 5 (M5): hand written layout where one is known, M3 elsewhere - */
  acc_t m5fb, m5elf;
  memset(&m5fb,0,sizeof(m5fb)); memset(&m5elf,0,sizeof(m5elf));
  m5fb.lit  = sopen(outdir,"m5/fallback.bin");
  m5elf.lit = sopen(outdir,"m5/fallback_elf.bin");
  u64 m5_pub_hit=0UL, m5_pub_miss=0UL, m5_recs=0UL, m5_elf_recs=0UL;
  {
    rec_t r; u64 off=16UL;
    while( next_rec(&off,&r) ){
      if( !r.data_len ) continue;
      layout_t * L = pick_layout(r.owner,r.data_len);
      grp_t * ga = gt_get(r.owner,r.data_len);
      grp_t * g = r.data_len<=maxlen ? ga : NULL;
      if( L ){ m5_record(L,g,r.data,r.data_len,&L->m5); m5_recs++; }
      else {
        /* attribute the fallback's bytes to the group, so that step 4 can see
           which programs are worth declaring next */
        acc_t * a = is_loader(r.owner) ? &m5elf : &m5fb;
        if( a==&m5elf ) m5_elf_recs++;
        u64 l0=a->lit_b, z0=a->zero_b, k0=a->keyref_b;
        m3_record(g,r.data,r.data_len,a,&m5_pub_hit,&m5_pub_miss);
        grp_t * go = gt_get(r.owner,(u64)-1);   /* the owner-only bucket */
        for( int k=0;k<2;k++ ){
          grp_t * t = k ? go : ga;
          t->fb_raw  += r.data_len;
          t->fb_zero += a->zero_b-z0;
          t->fb_kref += a->keyref_b-k0;
          t->fb_lit  += a->lit_b-l0;
        }
      }
    }
  }
  for( u64 i=0UL;i<NLAYOUT;i++ ){ fclose(layouts[i].m5.lit.f); fclose(layouts[i].m3.lit.f); }
  fclose(m5fb.lit.f); fclose(m5elf.lit.f);

  for( u64 gi=0UL; gi<ntemplated; gi++ ){
    grp_t * g=cand[gi]; layout_t * L=(layout_t*)g->lay;
    if( !L ) continue;
    for( u64 j=0UL;j<g->nslots;j++ ) if( g->field[j].ftype==FT_U64 && g->field[j].m5_used ){
      slot_t * sl=g->field+j;
      double h=card_entropy(&sl->card);
      double w=(double)sl->m5_used;
      double cb=w*(h>=0.0?h:64.0);
      double vb=slot_varint_bits(sl,g->cnt)*w/(double)g->cnt;
      int nu = !sl->card.over && (double)sl->card.cnt >= 0.5*(double)g->cnt;
      L->m5_u64_bits     += cb;
      L->m5_u64_bits_rob += nu ? vb : cb;
    }
  }

  /* per layout costs under both models.  The literal streams are on disk
     (outdir/m3lay/<name>.bin and outdir/m5/<name>.bin) so their real zstd
     size closes the comparison.  Every cost is reported twice: as measured,
     and with the sample bounded order-0 alphabets of near unique fields
     replaced by a varint (see the sampling exposure note under M3). */
  double m5_bits=0.0, m5_kref_split=0.0, m5_kref_shared=0.0;
  double m5_bits_rob=0.0, m3lay_bits_rob=0.0;
  u64 m5_zero=0UL, m5_keyref_b=0UL, m5_lit_b=0UL, m5_gen_b=0UL, m5_raw=0UL, m5_refs=0UL;
  double m3lay_bits=0.0; u64 m3lay_lit_b=0UL;
  double lay_m3_bits[NLAYOUT], lay_m5_bits[NLAYOUT];
  double lay_m3_bits_rob[NLAYOUT], lay_m5_bits_rob[NLAYOUT];
  for( u64 i=0UL;i<NLAYOUT;i++ ){
    layout_t * L=layouts+i;

    /* declared fields, plus the learned template slots the layout left alone */
    double num     = layout_field_bits(L->f,L->nf,L->fs,0);
    double num_rob = layout_field_bits(L->f,L->nf,L->fs,1);
    if( L->f_alt ){
      num     += layout_field_bits(L->f_alt,L->nf_alt,L->fs_alt,0);
      num_rob += layout_field_bits(L->f_alt,L->nf_alt,L->fs_alt,1);
    }
    double rest = bin_bits(L->m5_pub_hit,L->m5_pub_miss) + dec_bits(&L->tlv_pk.dec)
                + dec_bits(&L->m5.gdec);
    double ht=card_entropy(&L->tlv_type), hl=card_entropy(&L->tlv_len);
    if( L->tlv_n ) rest += (double)L->tlv_n*((ht>=0.0?ht:16.0)+(hl>=0.0?hl:16.0));
    double kshared = kref_entropy2(L->m5.kv,L->m5.kn,L->m5.gkv,L->m5.gkn);
    m5_kref_shared += kshared;
    lay_m5_bits[i]     = num     + rest + kshared + L->m5_u64_bits;
    lay_m5_bits_rob[i] = num_rob + rest + kshared + L->m5_u64_bits_rob;
    m5_bits += lay_m5_bits[i]; m5_bits_rob += lay_m5_bits_rob[i];

    /* the same references, coded as one stream per declared field instead */
    double split=kref_entropy_bits(L->m5.gkv,L->m5.gkn);
    for( u64 j=0UL;j<L->nf;j++ ) split += kref_entropy_bits(L->fs[j].kv,L->fs[j].kn);
    if( L->f_alt ) for( u64 j=0UL;j<L->nf_alt;j++ ) split += kref_entropy_bits(L->fs_alt[j].kv,L->fs_alt[j].kn);
    split += kref_entropy_bits(L->tlv_pk.kv,L->tlv_pk.kn);
    m5_kref_split += split;

    m5_zero+=L->m5.zero_b; m5_keyref_b+=L->m5.keyref_b; m5_lit_b+=L->m5.lit_b;
    m5_gen_b+=L->m5.gen_b; m5_raw+=L->raw; m5_refs += L->m5.kn+L->m5.gkn;

    double b3rest = bin_bits(L->m3_pub_hit,L->m3_pub_miss) + dec_bits(&L->m3.gdec)
                  + kref_entropy2(L->m3.kv,L->m3.kn,L->m3.gkv,L->m3.gkn);
    lay_m3_bits[i]     = L->m3_u64_bits     + b3rest;
    lay_m3_bits_rob[i] = L->m3_u64_bits_rob + b3rest;
    m3lay_bits += lay_m3_bits[i]; m3lay_bits_rob += lay_m3_bits_rob[i];
    m3lay_lit_b += L->m3.lit_b;
  }
  /* records without a layout keep the M3 model, including its template bits */
  double m5_fb_u64_bits = m3_u64_bits;
  for( u64 gi=0UL; gi<ntemplated; gi++ ) if( cand[gi]->lay ){
    grp_t * g=cand[gi];
    for( u64 j=0UL;j<g->nslots;j++ ) if( g->field[j].ftype==FT_U64 ){
      double h=card_entropy(&g->field[j].card);
      m5_fb_u64_bits -= (double)g->cnt*(h>=0.0?h:64.0);
    }
  }
  double m5_fb_bits = m5_fb_u64_bits + bin_bits(m5_pub_hit,m5_pub_miss)
                    + dec_bits(&m5fb.gdec)  + kref_entropy2(m5fb.kv,m5fb.kn,m5fb.gkv,m5fb.gkn)
                    + dec_bits(&m5elf.gdec) + kref_entropy2(m5elf.kv,m5elf.kn,m5elf.gkv,m5elf.gkn);
  double m5_total_bits = m5_bits + m5_fb_bits;
  u64    m5_total_lit  = m5_lit_b + m5fb.lit_b + m5elf.lit_b;

  /* machine readable per layout table for src/m5report.py */
  {
    char p[4096]; snprintf(p,sizeof(p),"%s/m5_layouts.csv",outdir);
    FILE * f=fopen(p,"w");
    if( f ){
      fprintf(f,"layout,records,raw_bytes,m3_bits,m3_lit_bytes,m5_bits,m5_lit_bytes,m3_bits_rob,m5_bits_rob\n");
      for( u64 i=0UL;i<NLAYOUT;i++ ){
        layout_t * L=layouts+i;
        fprintf(f,"%s,%lu,%lu,%.0f,%lu,%.0f,%lu,%.0f,%.0f\n",L->name,L->cnt+L->cnt_alt,L->raw,
                lay_m3_bits[i],L->m3.lit_b,lay_m5_bits[i],L->m5.lit_b,
                lay_m3_bits_rob[i],lay_m5_bits_rob[i]);
      }
      fprintf(f,"fallback,%lu,%lu,%.0f,%lu,%.0f,%lu,%.0f,%.0f\n",
              n-m5_recs-m5_elf_recs,data_bytes-m5_raw-m5elf.gen_b,
              0.0,0UL,m5_fb_bits,m5fb.lit_b,0.0,m5_fb_bits);
      fprintf(f,"fallback_elf,%lu,%lu,%.0f,%lu,%.0f,%lu,%.0f,%.0f\n",
              m5_elf_recs,m5elf.gen_b,0.0,0UL,0.0,m5elf.lit_b,0.0,0.0);
      fclose(f);
    }
  }

  /* ================================================================= */
  printf("oracle: %s (%lu records, %lu B data, %lu groups, %lu/%lu templated covering %lu records)\n",
         argv[2],n,data_bytes,gt_cnt,ntemplated,ncand,templated_records);

  printf("\nM1 -- metadata (data untouched, still zstd -- see corpsplit/bench_baseline.sh):\n");
  printf("  H(owner)                 %7.3f bits\n",h_owner);
  printf("  H(data_len | owner)      %7.3f bits\n",h_dlen);
  printf("  H(lamports | owner,len)  %7.3f bits\n",h_lam);
  printf("  H(rent_epoch| owner,len) %7.3f bits\n",h_rent);
  printf("  H(executable| owner,len) %7.3f bits\n",h_exe);
  printf("  pubkey (sorted set floor)%7.3f bits\n",pubkey_bits);
  printf("  metadata total           %7.3f bits = %.2f B/account\n",meta_bits,meta_bits/8.0);
  printf("  metadata + pubkey        %7.3f bits = %.2f B/account (x%lu = %.2f GiB)\n",
         meta_bits+pubkey_bits,(meta_bits+pubkey_bits)/8.0,n, (meta_bits+pubkey_bits)/8.0*(double)n/1073741824.0);

  printf("\nM2 -- ZERO32/KEYREF/LITERAL, no schema:\n");
  printf("  zero bytes    : %lu (%.2f%% of data)\n",m2_zero,100.0*(double)m2_zero/(double)(data_bytes?data_bytes:1));
  printf("  keyref bytes  : %lu (%.2f%% of data), %lu refs, H(ref)=%.2f bits total = %.1f MiB\n",
         m2_keyref_b,100.0*(double)m2_keyref_b/(double)(data_bytes?data_bytes:1),m2kn,m2_keyref_bits,m2_keyref_bits/8.0/1048576.0);
  printf("  literal bytes : %lu (%.2f%% of data) -> m2_literal.bin, zstd it\n",
         m2_lit_b,100.0*(double)m2_lit_b/(double)(data_bytes?data_bytes:1));
  printf("  token decision overhead: %.1f MiB (H over {zero,keyref,miss} decisions)\n",m2_dec_bits/8.0/1048576.0);
  printf("  M2 data total (excl. literal's own zstd size): %.1f MiB\n",(m2_keyref_bits+m2_dec_bits)/8.0/1048576.0);

  printf("\nM3 -- M2 + per (owner,len) templates (top %lu groups, len<=%lu):\n",topn,maxlen);
  u64 tot_const=0UL,tot_u64b=0UL,tot_pubb=0UL;
  for( u64 gi=0UL;gi<ntemplated;gi++ ){ tot_const+=cand[gi]->const_bytes; tot_u64b+=cand[gi]->u64_bytes; tot_pubb+=cand[gi]->pubkey_bytes; }
  printf("  const bytes freed     : %lu (%.2f%% of data)\n",tot_const,100.0*(double)tot_const/(double)(data_bytes?data_bytes:1));
  printf("  template u64 fields   : %lu raw bytes -> %.1f MiB (H per slot)\n",tot_u64b,m3_u64_bits/8.0/1048576.0);
  printf("  template pubkey fields: %lu raw bytes, hit %lu / miss %lu (%.2f%% hit), decision cost %.2f MiB\n",
         tot_pubb,pub_hit,pub_miss,100.0*(double)pub_hit/(double)(pub_hit+pub_miss?pub_hit+pub_miss:1),pub_bit/8.0/1048576.0);
  printf("  residual zero bytes   : %lu (%.2f%% of data)\n",m3_zero,100.0*(double)m3_zero/(double)(data_bytes?data_bytes:1));
  printf("  residual keyref bytes : %lu (%.2f%% of data), %lu refs, H(ref)=%.1f MiB\n",
         m3_keyref_b,100.0*(double)m3_keyref_b/(double)(data_bytes?data_bytes:1),m3kn,m3_keyref_bits/8.0/1048576.0);
  printf("  residual literal bytes: %lu (%.2f%% of data) -> m3_literal.bin, zstd it\n",
         m3_lit_b,100.0*(double)m3_lit_b/(double)(data_bytes?data_bytes:1));
  printf("  residual decision overhead: %.1f MiB\n",m3_dec_bits/8.0/1048576.0);
  printf("  M3 data total (excl. literal's own zstd size): %.1f MiB\n",
         (m3_keyref_bits+m3_dec_bits+m3_u64_bits+pub_bit)/8.0/1048576.0);

  /* How much of M3's u64 template cost is an artifact of the sample?  A slot's
     order-0 entropy is bounded by log2(group size), so a slot that is unique
     per record -- a hash, a blockhash, a signature -- is charged ~log2(cnt)
     bits here and would be charged ~64 in the full snapshot.  Report the
     exposure rather than guessing at it. */
  { double susp_bits=0.0, susp_raw=0.0, susp_var=0.0, other=0.0;
    u64 susp_slots=0UL, all_slots=0UL;
    for( u64 gi=0UL; gi<ntemplated; gi++ ){
      grp_t * g=cand[gi];
      for( u64 j=0UL;j<g->nslots;j++ ) if( g->field[j].ftype==FT_U64 ){
        slot_t * sl=g->field+j;
        all_slots++;
        double h=card_entropy(&sl->card);
        double cb=(double)g->cnt*(h>=0.0?h:64.0);
        if( sl->card.over || (double)sl->card.cnt < 0.5*(double)g->cnt ){ other += cb; continue; }
        susp_slots++;
        susp_bits += cb;
        susp_var  += slot_varint_bits(sl,g->cnt);
        susp_raw  += (double)g->cnt*64.0;
      }
    }
    printf("  sampling exposure: %lu of %lu u64 slots are near unique per record in the sample,\n",susp_slots,all_slots);
    printf("    so their order-0 alphabet is bounded by the sample size and not by the field:\n");
    printf("    charged above %.1f MiB, as varints %.1f MiB, at 64 bits/slot %.1f MiB\n",
           susp_bits/8.0/1048576.0,susp_var/8.0/1048576.0,susp_raw/8.0/1048576.0);
    printf("    -> M3's u64 term is %.1f MiB as measured and %.1f MiB with those slots as varints\n",
           m3_u64_bits/8.0/1048576.0,(other+susp_var)/8.0/1048576.0);
  }

  printf("\nM4a -- sorted order, whole record XORed vs previous, no dictionary:\n");
  printf("  zero bytes (post-XOR) : %lu (%.2f%% of data)\n",m4_zero,100.0*(double)m4_zero/(double)(data_bytes?data_bytes:1));
  printf("  literal bytes (XOR'd) : %lu (%.2f%% of data) -> m4_literal.bin, zstd it\n",
         m4_lit_b,100.0*(double)m4_lit_b/(double)(data_bytes?data_bytes:1));
  printf("  decision overhead     : %.1f MiB\n",m4_dec_bits/8.0/1048576.0);

  printf("\nM4b -- M3 + sorted order + REPEAT token (dictionary kept):\n");
  printf("  zero bytes            : %lu (%.2f%% of data)\n",m4b_zero,100.0*(double)m4b_zero/(double)(data_bytes?data_bytes:1));
  printf("  repeat bytes          : %lu (%.2f%% of data)\n",m4b_rep_b,100.0*(double)m4b_rep_b/(double)(data_bytes?data_bytes:1));
  printf("  keyref bytes          : %lu (%.2f%% of data), %lu refs, H(ref)=%.1f MiB\n",
         m4b_keyref_b,100.0*(double)m4b_keyref_b/(double)(data_bytes?data_bytes:1),m4bkn,m4b_keyref_bits/8.0/1048576.0);
  printf("  template pubkey fields: repeat %lu / keyref %lu / escape %lu (%.2f%% repeat)\n",
         m4b_pubrep,m4b_pubhit,m4b_pubmiss,
         100.0*(double)m4b_pubrep/(double)(m4b_pubrep+m4b_pubhit+m4b_pubmiss?m4b_pubrep+m4b_pubhit+m4b_pubmiss:1));
  printf("  template u64 fields    : %.1f MiB (per field min(raw,delta); delta-always would be %.1f MiB, raw-always %.1f MiB)\n",
         m4b_u64_bits/8.0/1048576.0,m4b_u64_delta_only/8.0/1048576.0,m3_u64_bits/8.0/1048576.0);
  printf("    delta beats raw on %lu of %lu u64 slots\n",m4b_u64_delta_wins,m4b_u64_slots);
  printf("  literal bytes          : %lu (%.2f%% of data) -> m4b_literal.bin, zstd it\n",
         m4b_lit_b,100.0*(double)m4b_lit_b/(double)(data_bytes?data_bytes:1));
  printf("  decision overhead      : %.1f MiB\n",m4b_dec_bits/8.0/1048576.0);
  printf("  M4b data total (excl. literal's own zstd size): %.1f MiB\n",
         (m4b_keyref_bits+m4b_dec_bits+m4b_u64_bits)/8.0/1048576.0);

  printf("\nM5 -- M3 + hand written layouts for known programs (PLAN.md step 6):\n");
  printf("  layouts               : %lu, covering %lu records (%.2f%% of the corpus), %lu B data (%.2f%% of data)\n",
         NLAYOUT,m5_recs,100.0*(double)m5_recs/(double)(n?n:1),m5_raw,100.0*(double)m5_raw/(double)(data_bytes?data_bytes:1));
  printf("  zero bytes            : %lu (%.2f%% of the covered data)\n",
         m5_zero,100.0*(double)m5_zero/(double)(m5_raw?m5_raw:1));
  printf("  keyref bytes          : %lu (%.2f%% of the covered data), %lu refs\n",
         m5_keyref_b,100.0*(double)m5_keyref_b/(double)(m5_raw?m5_raw:1),m5_refs);
  printf("  literal bytes         : %lu (%.2f%% of the covered data) -> %s/m5/*.bin\n",
         m5_lit_b,100.0*(double)m5_lit_b/(double)(m5_raw?m5_raw:1),outdir);
  printf("  bytes left to the generic walker inside covered records: %lu (%.2f%%)\n",
         m5_gen_b,100.0*(double)m5_gen_b/(double)(m5_raw?m5_raw:1));
  printf("  analytic bits, covered: %.1f MiB (one keyref stream per layout)\n",m5_bits/8.0/1048576.0);
  printf("                          %.1f MiB (one keyref stream per declared field: %.1f vs %.1f MiB of reference entropy)\n",
         (m5_bits-m5_kref_shared+m5_kref_split)/8.0/1048576.0,
         m5_kref_split/8.0/1048576.0,m5_kref_shared/8.0/1048576.0);
  printf("  analytic bits, rest   : %.1f MiB (M3 model on %lu records, of which %lu are\n",
         m5_fb_bits/8.0/1048576.0,n-m5_recs,m5_elf_recs);
  printf("                          loader owned: %lu B of data -> %lu literal bytes in m5/fallback_elf.bin)\n",
         m5elf.gen_b,m5elf.lit_b);
  printf("  M5 data total (excl. literal's own zstd size): %.1f MiB, literal %lu B\n",
         m5_total_bits/8.0/1048576.0,m5_total_lit);
  printf("  same records under M3 : analytic %.1f MiB, literal %lu B\n",
         m3lay_bits/8.0/1048576.0,m3lay_lit_b);
  printf("  -> covered records, M5 vs M3 analytic: %.1f vs %.1f MiB; literal %lu vs %lu B\n",
         m5_bits/8.0/1048576.0,m3lay_bits/8.0/1048576.0,m5_lit_b,m3lay_lit_b);
  printf("     same, with near unique numeric fields charged as varints instead of by their\n");
  printf("     sample bounded alphabet: M5 %.1f vs M3 %.1f MiB analytic\n",
         m5_bits_rob/8.0/1048576.0,m3lay_bits_rob/8.0/1048576.0);
  printf("  per layout CSV        : %s/m5_layouts.csv (price the literal streams with\n",outdir);
  printf("                          src/bench_baseline.sh %s/m5 and %s/m3lay, then src/m5report.py)\n",outdir,outdir);

  printf("\n  per layout (records, B/rec, and what the declared fields resolved to):\n");
  for( u64 i=0UL;i<NLAYOUT;i++ ){
    layout_t * L=layouts+i;
    if( !L->cnt && !L->cnt_alt ) continue;
    printf("  %-12s cnt=%-9lu raw=%-12lu zero=%-11lu keyref=%-11lu lit=%-10lu gen=%lu\n",
           L->name,L->cnt+L->cnt_alt,L->raw,L->m5.zero_b,L->m5.keyref_b,L->m5.lit_b,L->m5.gen_b);
    for( int alt=0; alt<2; alt++ ){
      lf_t const * f = alt?L->f_alt:L->f; u64 nf=alt?L->nf_alt:L->nf; fstat_t * fs=alt?L->fs_alt:L->fs;
      if( !f ) continue;
      if( alt ) printf("    (mint variant, account_type==1)\n");
      for( u64 j=0UL;j<nf;j++ ){
        fstat_t * s=fs+j;
        if( !s->n ) continue;
        if( f[j].kind==LK_PUBKEY || f[j].kind==LK_ARR_PUBKEY ){
          u64 t=s->dec.zero+s->dec.keyref+s->dec.miss;
          printf("    off %5u %-6s n=%-10lu zero %5.1f%%  keyref %5.1f%%  escape %5.1f%%  dec %.2f MiB\n",
                 f[j].off,f[j].kind==LK_PUBKEY?"pubkey":"pk[]",s->n,
                 100.0*(double)s->dec.zero/(double)(t?t:1),
                 100.0*(double)s->dec.keyref/(double)(t?t:1),
                 100.0*(double)s->dec.miss/(double)(t?t:1),
                 dec_bits(&s->dec)/8.0/1048576.0);
        } else if( f[j].kind==LK_NUM ){
          int varint=0; double b=fs_num_bits(s,&varint);
          double h=card_entropy(&s->card);
          printf("    off %5u num%-3u n=%-10lu %s %6.3f bits/rec  (alphabet %s)\n",
                 f[j].off,f[j].w,s->n,varint?"varint":"order0",b/(double)(s->n?s->n:1),
                 h>=0.0?"measured":">64k");
        } else if( f[j].kind==LK_ZPAD ){
          u64 t=s->dec.zero+s->dec.miss;
          printf("    off %5u zpad%-2u n=%-10lu zero %5.1f%%\n",
                 f[j].off,f[j].w,s->n,100.0*(double)s->dec.zero/(double)(t?t:1));
        }
      }
    }
    if( L->tlv_n ){
      u64 t=L->tlv_pk.dec.zero+L->tlv_pk.dec.keyref+L->tlv_pk.dec.miss;
      printf("    TLV: %lu entries, %lu distinct types, %lu entries of a type with no declared pubkey;\n",
             L->tlv_n,L->tlv_type.over?0UL:L->tlv_type.cnt,L->tlv_unknown);
      printf("         declared pubkeys n=%lu keyref %5.1f%% zero %5.1f%% escape %5.1f%%\n",
             t,100.0*(double)L->tlv_pk.dec.keyref/(double)(t?t:1),
             100.0*(double)L->tlv_pk.dec.zero/(double)(t?t:1),
             100.0*(double)L->tlv_pk.dec.miss/(double)(t?t:1));
      /* which extensions are actually out there, so the table above can grow
         towards the ones that matter */
      if( !L->tlv_type.over ){
        printf("         top types:");
        for( int k=0;k<8;k++ ){
          u64 best=0UL, bv=0UL; int found=0;
          for( u64 q=0UL;q<L->tlv_type.cap;q++ ) if( L->tlv_type.t[q].used && L->tlv_type.t[q].c>best ){
            best=L->tlv_type.t[q].c; bv=L->tlv_type.t[q].v; found=1;
          }
          if( !found ) break;
          printf(" %lu:%lu%s",bv,best,ext_find((u32)bv)?"*":"");
          for( u64 q=0UL;q<L->tlv_type.cap;q++ ) if( L->tlv_type.t[q].used && L->tlv_type.t[q].v==bv ) L->tlv_type.t[q].used=0;
        }
        printf("   (* = pubkeys declared)\n");
      }
    }
  }

  printf("\nnext step: run bench_baseline.sh %s to price m2/m3/m4_literal.bin with real zstd,\n",outdir);
  printf("then add (metadata+pubkey) + keyref-entropy + template-field bits + zstd(literal) per model\n");
  printf("and compare against zstd -19 on account data (%s) per PLAN.md's gate.\n", "corpsplit's data.bin");

  /* Where the residual lives (PLAN.md step 4): the groups no layout covers,
     ranked by the literal bytes they contribute, which is what the fallback
     actually costs. */
  {
    for( int by_owner=1; by_owner>=0; by_owner-- ){
      grp_t ** fb = malloc(gt_cnt*sizeof(grp_t*)); u64 nfb=0UL;
      for( u64 i=0UL;i<gt_cap;i++ )
        if( gt[i].used && gt[i].fb_raw && (gt[i].len==(u64)-1)==(by_owner!=0) ) fb[nfb++]=gt+i;
      for( u64 i=0UL;i<nfb;i++ ) for( u64 j=i+1UL;j<nfb;j++ )
        if( fb[j]->fb_lit > fb[i]->fb_lit ){ grp_t * t=fb[i]; fb[i]=fb[j]; fb[j]=t; }
      printf("\nM5 fallback: where the uncovered residual lives, top 16 %s by literal bytes\n",
             by_owner?"owners":"(owner,len) groups");
      printf("  %-46s %9s %10s %13s %13s %6s %6s %6s\n",
             "owner",by_owner?"lens":"len","records","raw B","literal B","zero%","kref%","lit%");
      for( u64 i=0UL;i<nfb && i<16UL;i++ ){
        grp_t * g=fb[i]; char b[64]; b58(g->owner,b);
        if( by_owner ) printf("  %-46s %9s",b,"*");
        else           printf("  %-46s %9lu",b,g->len);
        printf(" %10lu %13lu %13lu %6.2f %6.2f %6.2f\n",
               g->cnt,g->fb_raw,g->fb_lit,
               100.0*(double)g->fb_zero/(double)g->fb_raw,
               100.0*(double)g->fb_kref/(double)g->fb_raw,
               100.0*(double)g->fb_lit /(double)g->fb_raw);
      }
      free(fb);
    }
  }

  /* per-slot map of the biggest few groups: what the template actually
     decided at each offset, and the dictionary hit rate that drove it.  This
     is where a hand written layout for a known program would differ from what
     the statistics guess. */
  printf("\nper-slot template map (top 3 groups; hit = 32B window at this offset is a known pubkey):\n");
  for( u64 gi=0UL; gi<ntemplated && gi<3UL; gi++ ){
    grp_t * g=cand[gi]; char s[64]; b58(g->owner,s);
    printf("  [%lu] %s len=%lu cnt=%lu\n",gi,s,g->len,g->cnt);
    for( u64 j=0UL;j<g->nslots;j++ ){
      slot_t * sl=g->field+j;
      char const * t = sl->ftype==FT_CONST?"const":sl->ftype==FT_PUBKEY?"PUBKEY":
                       sl->ftype==FT_PUBKEY_CONT?"  ^":sl->ftype==FT_U64?"u64":"free";
      double hr = sl->tot ? 100.0*(double)sl->hit/(double)sl->tot : -1.0;
      double he = card_entropy(&sl->card);
      printf("    off %4lu  %-6s  hit ",j*FLD8,t);
      if( hr>=0.0 ) printf("%6.2f%%",hr); else printf("     - ");
      if( he>=0.0 ) printf("  H=%.2f bits\n",he); else printf("  H=>64k\n");
    }
  }

  /* per-group summary of the biggest templated groups */
  printf("\ntop templated groups (owner, len, cnt, const/u64/pubkey/free bytes per record):\n");
  for( u64 gi=0UL; gi<ntemplated && gi<32UL; gi++ ){
    grp_t * g=cand[gi]; char s[64]; b58(g->owner,s);
    u64 free_bytes=0UL; for(u64 si=0UL;si<g->nspans;si++) free_bytes+=g->spans[si].len;
    printf("  %-46s len=%-6lu cnt=%-10lu const=%-4lu u64=%-4lu pubkey=%-4lu free=%-4lu\n",
           s,g->len,g->cnt, g->const_bytes/g->cnt, g->u64_bytes/g->cnt, g->pubkey_bytes/g->cnt, free_bytes);
  }
  return 0;
}
