# Fork-owned build rules (Ternary Bonsai 2).  Included once from the upstream
# Makefile right after the platform object lists are defined and before any
# target that expands $(CORE_OBJS) as a prerequisite.  Keep everything fork
# specific here so the upstream Makefile diff stays at one line.

# Rules below must not steal the upstream default goal ("all").
FORK_SAVED_DEFAULT_GOAL := $(.DEFAULT_GOAL)

ifeq ($(UNAME_S),Darwin)
CORE_OBJS += ds4_metal_bonsai2.o
FORK_METAL_OBJS := ds4_metal_bonsai2.o

ds4_metal_bonsai2.o: ds4_metal_bonsai2.m ds4_metal_internal.h ds4_bonsai2_gpu.h ds4_gpu.h
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal_bonsai2.m

# ds4_metal.m includes the internal header; metal/bonsai2.metal is already in
# $(METAL_SRCS) through the upstream wildcard.
ds4_metal.o: ds4_metal_internal.h

tests/bonsai2/test_kernels.o: tests/bonsai2/test_kernels.c ds4_bonsai2_gpu.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/bonsai2/test_kernels.c

tests/bonsai2/test_kernels: tests/bonsai2/test_kernels.o ds4_metal.o ds4_metal_bonsai2.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

# Kernel unit tests against CPU references (no model needed).
.PHONY: test-bonsai2-kernels bench-bonsai2-kernels test-bonsai2-reference
test-bonsai2-kernels: tests/bonsai2/test_kernels
	./tests/bonsai2/test_kernels

bench-bonsai2-kernels: tests/bonsai2/test_kernels
	./tests/bonsai2/test_kernels --bench

# End-to-end greedy/logprob match against the PrismML reference (needs the
# model; BONSAI2_MODEL overrides the default gguf/ path).
test-bonsai2-reference: ds4
	./tests/bonsai2/check_reference.py $(if $(BONSAI2_MODEL),--model $(BONSAI2_MODEL))

# Plain Qwen3.8-27B against llama.cpp on the same GGUF (QWEN38_27B_MODEL
# overrides it).  Greedy tokens must match exactly.  The logprob tolerance is
# llama.cpp's error, not ds4's: its few-row Q4_K matmuls round activations to
# half, and a float64 recomputation of layer 0 agrees with ds4 to 2e-7 but
# with llama.cpp only to 3e-3, which reaches 0.1-0.35 in top-5 logprobs
# (docs/QWEN38_27B.md).
QWEN38_27B_MODEL ?= gguf/qwen38-27b/Qwen3.8-27B-Q4KM-Q8.gguf
# Quality against a reference model (docs/QWEN38_27B.md, Quality).
tests/qwen38_27b/kld.o: tests/qwen38_27b/kld.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/qwen38_27b/kld: tests/qwen38_27b/kld.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: test-qwen38-27b-reference
test-qwen38-27b-reference: ds4
	./tests/bonsai2/check_reference.py --model $(QWEN38_27B_MODEL) \
		--golden tests/qwen38_27b/golden.json --tol 0.4
endif

# ds4.c textually includes the fork engine code.
ds4.o ds4_cpu.o: ds4_bonsai2_shape.inc ds4_bonsai2_model.inc ds4_bonsai2.inc ds4_bonsai2_batch.inc ds4_bonsai2_gpu.h

.PHONY: fork-clean
clean: fork-clean
fork-clean:
	rm -f ds4_metal_bonsai2.o tests/bonsai2/test_kernels tests/bonsai2/test_kernels.o \
		tests/qwen38_27b/kld tests/qwen38_27b/kld.o

.DEFAULT_GOAL := $(FORK_SAVED_DEFAULT_GOAL)
