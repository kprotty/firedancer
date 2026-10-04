ifdef FD_HAS_ATOMIC
ifdef FD_HAS_ALLOCA

$(call add-hdrs,fd_accdb.h fd_accdb_base.h fd_accdb_cache.h fd_accdb_shmem.h)
$(call add-objs,fd_accdb fd_accdb_cache fd_accdb_shmem,fd_flamenco)
$(call add-objs,fd_accdb_tile,fd_disco) # TODO: MOVE TO DISCOF

$(call make-unit-test,test_accdb,test_accdb,fd_flamenco fd_ballet fd_util)
$(call run-unit-test,test_accdb)

$(call make-unit-test,test_accdb_cache,test_accdb_cache,fd_flamenco fd_ballet fd_util)
$(call run-unit-test,test_accdb_cache)

$(call make-unit-test,bench_accdb,bench_accdb,fd_flamenco fd_ballet fd_util)
$(call make-unit-test,bench_accdb_hotread,bench_accdb_hotread,fd_flamenco fd_ballet fd_util)
$(call make-unit-test,bench_accdb_txn,bench_accdb_txn,fd_flamenco fd_ballet fd_util)

ifdef FD_HAS_RACESAN
$(call make-unit-test,test_accdb_racesan,test_accdb_racesan,fd_flamenco fd_ballet fd_util)
endif

endif
endif

$(call add-hdrs,fd_zle.h)
$(call add-objs,fd_zle,fd_flamenco)

$(call make-unit-test,test_zle,test_zle,fd_flamenco fd_util)
$(call run-unit-test,test_zle)
$(call make-unit-test,bench_zle,bench_zle,fd_flamenco fd_util)

$(call add-hdrs,fd_zle2.h)
$(call add-objs,fd_zle2,fd_flamenco)

$(call make-unit-test,test_zle2,test_zle2,fd_flamenco fd_util)
$(call run-unit-test,test_zle2)
$(call make-unit-test,bench_zle2,bench_zle2,fd_flamenco fd_util)

$(call add-hdrs,fd_zle3.h)
$(call add-objs,fd_zle3,fd_flamenco)

$(call make-unit-test,test_zle3,test_zle3,fd_flamenco fd_util)
$(call run-unit-test,test_zle3)
$(call make-unit-test,bench_zle3,bench_zle3,fd_flamenco fd_util)

$(call add-hdrs,fd_zle4.h)
$(call add-objs,fd_zle4,fd_flamenco)

$(call add-hdrs,fd_zle5.h)
$(call add-objs,fd_zle5,fd_flamenco)

$(call make-unit-test,test_zle5,test_zle5,fd_flamenco fd_util)
$(call run-unit-test,test_zle5)
$(call make-unit-test,bench_zle5,bench_zle5,fd_flamenco fd_util)

$(call make-unit-test,test_zle4,test_zle4,fd_flamenco fd_util)
$(call run-unit-test,test_zle4)
$(call make-unit-test,bench_zle4,bench_zle4,fd_flamenco fd_util)

$(call add-hdrs,fd_zle6.h)
$(call add-objs,fd_zle6,fd_flamenco)

$(call make-unit-test,test_zle6,test_zle6,fd_flamenco fd_util)
$(call run-unit-test,test_zle6)
$(call make-unit-test,bench_zle6,bench_zle6,fd_flamenco fd_util)

$(call add-hdrs,fd_zle7.h)
$(call add-objs,fd_zle7,fd_flamenco)

$(call make-unit-test,test_zle7,test_zle7,fd_flamenco fd_util)
$(call run-unit-test,test_zle7)
$(call make-unit-test,bench_zle7_vget,bench_zle7_vget,fd_flamenco fd_util)
$(call make-unit-test,bench_zle7,bench_zle7,fd_flamenco fd_util)

$(call add-hdrs,fd_zle9.h)
$(call add-objs,fd_zle9,fd_flamenco)

$(call make-unit-test,test_zle9,test_zle9,fd_flamenco fd_util)
$(call run-unit-test,test_zle9)
$(call make-unit-test,bench_zle9,bench_zle9,fd_flamenco fd_util)
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle9_diff,fuzz_zle9_diff,fd_flamenco fd_util)
endif

$(call add-hdrs,fd_zle10.h)
$(call add-objs,fd_zle10,fd_flamenco)

$(call make-unit-test,test_zle10,test_zle10,fd_flamenco fd_util)
$(call run-unit-test,test_zle10)
$(call make-unit-test,bench_zle10,bench_zle10,fd_flamenco fd_util)
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle10_diff,fuzz_zle10_diff,fd_flamenco fd_util)
endif

$(call add-hdrs,fd_zle11.h)
$(call add-objs,fd_zle11,fd_flamenco)

$(call make-unit-test,test_zle11,test_zle11,fd_flamenco fd_util)
$(call run-unit-test,test_zle11)
$(call make-unit-test,bench_zle11,bench_zle11,fd_flamenco fd_util)
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle11_diff,fuzz_zle11_diff,fd_flamenco fd_util)
$(call make-fuzz-test,fuzz_zle11_decompress,fuzz_zle11_decompress,fd_flamenco fd_util)
endif

$(call add-hdrs,fd_zle12.h)
$(call add-objs,fd_zle12,fd_flamenco)

$(call make-unit-test,test_zle12,test_zle12,fd_flamenco fd_util)
$(call run-unit-test,test_zle12)
$(call make-unit-test,bench_zle12,bench_zle12,fd_flamenco fd_util)
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle12_diff,fuzz_zle12_diff,fd_flamenco fd_util)
endif

$(call add-hdrs,fd_zle13.h)
$(call add-objs,fd_zle13,fd_flamenco)

$(call make-unit-test,test_zle13,test_zle13,fd_flamenco fd_util)
$(call run-unit-test,test_zle13)
$(call make-unit-test,bench_zle13,bench_zle13,fd_flamenco fd_util)
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle13_diff,fuzz_zle13_diff,fd_flamenco fd_util)
endif

ifdef FD_HAS_AVX512
$(call add-hdrs,fd_zle14.h)
$(call add-objs,fd_zle14,fd_flamenco)

$(call make-unit-test,test_zle14,test_zle14,fd_flamenco fd_util)
$(call run-unit-test,test_zle14)
$(call make-unit-test,bench_zle14,bench_zle14,fd_flamenco fd_util)
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle14_diff,fuzz_zle14_diff,fd_flamenco fd_util)
$(call make-fuzz-test,fuzz_zle14_decompress,fuzz_zle14_decompress,fd_flamenco fd_util)
endif
endif

$(call add-hdrs,fd_zle8.h)
$(call add-objs,fd_zle8,fd_flamenco)

$(call make-unit-test,test_zle8,test_zle8,fd_flamenco fd_util)
$(call run-unit-test,test_zle8)
$(call make-unit-test,bench_zle8,bench_zle8,fd_flamenco fd_util)
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle8_diff,fuzz_zle8_diff,fd_flamenco fd_util)
$(call make-fuzz-test,fuzz_zle8_decompress,fuzz_zle8_decompress,fd_flamenco fd_util)
endif
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle7_diff,fuzz_zle7_diff,fd_flamenco fd_util)
$(call make-fuzz-test,fuzz_zle7_decompress,fuzz_zle7_decompress,fd_flamenco fd_util)
endif
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle6_diff,fuzz_zle6_diff,fd_flamenco fd_util)
$(call make-fuzz-test,fuzz_zle6_decompress,fuzz_zle6_decompress,fd_flamenco fd_util)
endif
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle4_diff,fuzz_zle4_diff,fd_flamenco fd_util)
$(call make-fuzz-test,fuzz_zle4_decompress,fuzz_zle4_decompress,fd_flamenco fd_util)
endif
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle3_diff,fuzz_zle3_diff,fd_flamenco fd_util)
$(call make-fuzz-test,fuzz_zle3_decompress,fuzz_zle3_decompress,fd_flamenco fd_util)
endif
ifdef FD_HAS_HOSTED
$(call make-fuzz-test,fuzz_zle_diff,fuzz_zle_diff,fd_flamenco fd_util)
$(call make-fuzz-test,fuzz_zle_decompress,fuzz_zle_decompress,fd_flamenco fd_util)
endif
