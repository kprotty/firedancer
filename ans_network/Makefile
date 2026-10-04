# ans_network - a neural network whose units are ANS encoders.
#
# Standalone.  Nothing here is wired into the Firedancer build; it
# shares no headers and no build machinery with the rest of the tree.

CC       ?= gcc
CFLAGS   ?= -O3 -march=native
CSTD     := -std=gnu11
WARN     := -Wall -Wextra
OPENMP   ?= -fopenmp
LDLIBS   := -lm

BIN      := ans_train ans_infer ans_test ans_bench_float ans_transformer
LIB      := ans.o ans_avx512.o ans_uabs.o model.o data.o
DATA_URL := https://ossci-datasets.s3.amazonaws.com/mnist
IDX      := train-images-idx3-ubyte train-labels-idx1-ubyte \
            t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte
DATA     := $(addprefix data/,$(IDX))

.PHONY: all clean check data demo
all: $(BIN)

%.o: %.c ans.h ans_avx512.h ans_uabs.h model.h data.h transformer.h
	$(CC) $(CSTD) $(CFLAGS) $(WARN) $(OPENMP) -c $< -o $@

ans_train: train.o $(LIB)
	$(CC) $(OPENMP) $^ $(LDLIBS) -o $@
ans_infer: infer.o $(LIB)
	$(CC) $(OPENMP) $^ $(LDLIBS) -o $@
ans_test: test.o $(LIB)
	$(CC) $(OPENMP) $^ $(LDLIBS) -o $@
ans_bench_float: bench_float.o $(LIB)
	$(CC) $(OPENMP) $^ $(LDLIBS) -o $@
ans_transformer: bench_transformer.o transformer.o $(LIB)
	$(CC) $(OPENMP) $^ $(LDLIBS) -o $@

# MNIST.  Without it everything still runs, on a synthetic set.
data: $(DATA)
$(DATA):
	@mkdir -p data
	@for f in $(IDX); do \
	  [ -f data/$$f ] || { echo "fetch $$f"; \
	    curl -sSfL -o data/$$f.gz $(DATA_URL)/$$f.gz && gunzip -f data/$$f.gz; }; \
	done

check: ans_test ans_transformer
	./ans_test
	./ans_transformer --check
	python3 check_ref.py

demo: all data
	./ans_train --hidden 256 --thr 4 --epochs 40
	./ans_infer

clean:
	rm -f $(BIN) *.o
