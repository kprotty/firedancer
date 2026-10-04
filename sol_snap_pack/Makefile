CC      := clang-19
CFLAGS  ?= -O2 -march=native -Wall -Wextra
BINS     = snapscan corpstat corpsplit corpgroup refscan oracle

all: $(BINS)

%: src/%.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(BINS)

.PHONY: all clean
