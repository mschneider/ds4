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
endif

# ds4.c textually includes the fork engine code.
ds4.o ds4_cpu.o: ds4_bonsai2_shape.inc ds4_bonsai2_model.inc ds4_bonsai2.inc ds4_bonsai2_batch.inc ds4_bonsai2_gpu.h

.PHONY: fork-clean
clean: fork-clean
fork-clean:
	rm -f ds4_metal_bonsai2.o tests/bonsai2/test_kernels tests/bonsai2/test_kernels.o

.DEFAULT_GOAL := $(FORK_SAVED_DEFAULT_GOAL)
