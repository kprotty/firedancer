#ifndef HEADER_fd_src_util_math_fd_rcp_h
#define HEADER_fd_src_util_math_fd_rcp_h

/* fd_rcp provides fast division and modulo by a runtime divisor d that
   is reused across many operations (e.g. hoisted out of a loop).
   mul==floor((2^64-1)/d) makes q=(x*mul)>>64 either x/d or one less,
   which a single compare against the remainder corrects.  This turns
   each division into two multiplies and a few ALU ops instead of a
   hardware divide.  Falls back to hardware division on targets without
   uint128.  Typical usage:

     fd_rcp_t rcp = fd_rcp( cnt );
     for( ulong i=0UL; i<n; i++ ) idx[ i ] = fd_rcp_mod( rcp, hash[ i ] ); */

#include "../bits/fd_bits.h"

struct fd_rcp {
  ulong mul;
  ulong d;
};

typedef struct fd_rcp fd_rcp_t;

FD_PROTOTYPES_BEGIN

/* fd_rcp returns the reciprocal of d.  d==0 is treated as d==1. */

FD_FN_CONST static inline fd_rcp_t
fd_rcp( ulong d ) {
  d = fd_ulong_max( d, 1UL );
  fd_rcp_t rcp = { .mul = ULONG_MAX/d, .d = d };
  return rcp;
}

/* fd_rcp_div returns x/d given rcp==fd_rcp( d ). */

FD_FN_CONST static inline ulong
fd_rcp_div( fd_rcp_t rcp,
            ulong    x ) {
# if FD_HAS_INT128
  ulong q = (ulong)( ( (uint128)x*(uint128)rcp.mul )>>64 );
  ulong r = x - q*rcp.d;
  return q + (ulong)( r>=rcp.d );
# else
  return x / rcp.d;
# endif
}

/* fd_rcp_mod returns x%d given rcp==fd_rcp( d ). */

FD_FN_CONST static inline ulong
fd_rcp_mod( fd_rcp_t rcp,
            ulong    x ) {
# if FD_HAS_INT128
  ulong q = (ulong)( ( (uint128)x*(uint128)rcp.mul )>>64 );
  ulong r = x - q*rcp.d;
  return fd_ulong_if( r>=rcp.d, r-rcp.d, r );
# else
  return x % rcp.d;
# endif
}

FD_PROTOTYPES_END

#endif /* HEADER_fd_src_util_math_fd_rcp_h */
