/* Reader side of the expert pack (ds4_expert_pack.inc) against a model and
 * its pack written by gguf-tools/expert_pack.py. Driven by
 * tests/test_expert_pack.py, which builds a synthetic GGUF.
 *
 * usage: test_expert_pack MODEL TABLE      (expects the pack to be used)
 *        test_expert_pack MODEL --rejected (expects the pack to be refused)
 * TABLE lists "model_off expert_bytes n_expert" for every expert tensor in
 * gate, up, down order per layer. */
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

typedef struct {
    uint64_t offset;
    uint64_t len;
    uint8_t *dst;
    uint64_t read_bytes;
    double ms;
    int ok;
    uint32_t pack;
    uint32_t pack_tails;
    uint8_t *tail_dst[2];
    uint64_t tail_len[2];
} ds4_gpu_stream_expert_pread_task;

static double ds4_gpu_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

#include "../ds4_expert_pack.inc"

static int g_model;

static void model_read(uint8_t *dst, uint64_t len, uint64_t off) {
    if (!ds4_expert_pack_pread_full(g_model, dst, len, off)) {
        fprintf(stderr, "model read failed\n");
        exit(1);
    }
}

/* Runs tasks the way ds4_metal.m does and checks each destination. */
static int run_and_check(ds4_gpu_stream_expert_pread_task *t, uint32_t n,
                         uint32_t want_reads, int coalesce_twice) {
    uint64_t model_off[16];
    for (uint32_t i = 0; i < n; i++) model_off[i] = t[i].offset;
    if (coalesce_twice) ds4_expert_pack_coalesce(t, n);
    ds4_expert_pack_coalesce(t, n);
    uint32_t reads = 0;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (t[i].pack) {
            t[i].ok = ds4_expert_pack_run_task(&t[i]);
            if (t[i].pack == DS4_EXPERT_PACK_TASK_READ) reads++;
        } else {
            model_read(t[i].dst, t[i].len, t[i].offset);
            t[i].read_bytes = t[i].len;
            t[i].ok = 1;
            reads++;
        }
        if (!t[i].ok) return 0;
        bytes += t[i].read_bytes;
    }
    uint64_t want_bytes = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t *ref = malloc(t[i].len);
        model_read(ref, t[i].len, model_off[i]);
        const int same = memcmp(ref, t[i].dst, t[i].len) == 0;
        free(ref);
        if (!same) {
            fprintf(stderr, "task %u: bytes differ from the model\n", i);
            return 0;
        }
        want_bytes += t[i].len;
    }
    if (bytes != want_bytes) {
        fprintf(stderr, "read %llu bytes, want %llu\n",
                (unsigned long long)bytes, (unsigned long long)want_bytes);
        return 0;
    }
    if (reads != want_reads) {
        fprintf(stderr, "%u reads, want %u\n", reads, want_reads);
        return 0;
    }
    return 1;
}

typedef struct { uint64_t off, bytes; uint32_t n; } tensor;

static ds4_gpu_stream_expert_pread_task task_for(const tensor *x, uint32_t e,
                                                 uint64_t inner, uint64_t len) {
    return (ds4_gpu_stream_expert_pread_task) {
        .offset = x->off + e * x->bytes + inner,
        .len = len,
        .dst = malloc(len),
    };
}

static void free_tasks(ds4_gpu_stream_expert_pread_task *t, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) free(t[i].dst);
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    g_model = open(argv[1], O_RDONLY);
    if (g_model < 0) return 2;
    ds4_expert_pack_attach(g_model);
    if (!strcmp(argv[2], "--rejected")) {
        if (g_expert_pack_fd >= 0) {
            fprintf(stderr, "a bad pack was accepted\n");
            return 1;
        }
        printf("rejected as expected\n");
        return 0;
    }
    if (g_expert_pack_fd < 0) {
        fprintf(stderr, "the pack was not attached\n");
        return 1;
    }

    FILE *f = fopen(argv[2], "r");
    tensor tens[64];
    uint32_t n_tens = 0;
    unsigned long long off, bytes;
    unsigned n;
    while (n_tens < 64 && fscanf(f, "%llu %llu %u", &off, &bytes, &n) == 3) {
        tens[n_tens++] = (tensor) { off, bytes, n };
    }
    fclose(f);
    if (n_tens == 0 || n_tens % 3) return 2;

    int ok = 1;
    for (uint32_t l = 0; l < n_tens / 3; l++) {
        const tensor *g = &tens[3 * l], *u = &tens[3 * l + 1], *d = &tens[3 * l + 2];
        for (uint32_t e = 0; e < g->n; e++) {
            /* gate, up, down of one expert: one read. */
            ds4_gpu_stream_expert_pread_task t[3] = {
                task_for(g, e, 0, g->bytes), task_for(u, e, 0, u->bytes), task_for(d, e, 0, d->bytes),
            };
            ok &= run_and_check(t, 3, 1, 0);
            free_tasks(t, 3);
        }
        /* Two experts back to back: one read each. */
        ds4_gpu_stream_expert_pread_task two[6] = {
            task_for(g, 1, 0, g->bytes), task_for(u, 1, 0, u->bytes), task_for(d, 1, 0, d->bytes),
            task_for(g, 2, 0, g->bytes), task_for(u, 2, 0, u->bytes), task_for(d, 2, 0, d->bytes),
        };
        ok &= run_and_check(two, 6, 2, 0);
        free_tasks(two, 6);
        /* Prefill order: gate and up merge; down reads alone. */
        ds4_gpu_stream_expert_pread_task gu[3] = {
            task_for(g, 3, 0, g->bytes), task_for(u, 3, 0, u->bytes), task_for(d, 0, 0, d->bytes),
        };
        ok &= run_and_check(gu, 3, 2, 0);
        free_tasks(gu, 3);
        /* A slice inside one expert is redirected; a range over two is not. */
        ds4_gpu_stream_expert_pread_task part[2] = {
            task_for(d, 1, 7, d->bytes - 9), task_for(g, 0, g->bytes / 2, g->bytes),
        };
        ok &= run_and_check(part, 2, 2, 0);
        ok &= part[0].pack == DS4_EXPERT_PACK_TASK_READ && part[1].pack == 0;
        free_tasks(part, 2);
        /* A whole tensor (qwen4_stream_read_layer) reads the model. */
        ds4_gpu_stream_expert_pread_task whole[1] = { task_for(g, 0, 0, g->bytes * g->n) };
        ok &= run_and_check(whole, 1, 1, 0);
        ok &= whole[0].pack == 0;
        free_tasks(whole, 1);
    }
    /* Coalescing twice (pool fallback path) must not change the result. */
    ds4_gpu_stream_expert_pread_task again[3] = {
        task_for(&tens[0], 0, 0, tens[0].bytes), task_for(&tens[1], 0, 0, tens[1].bytes),
        task_for(&tens[2], 0, 0, tens[2].bytes),
    };
    ok &= run_and_check(again, 3, 1, 1);
    free_tasks(again, 3);

    uint64_t pack_off;
    ok &= !ds4_expert_pack_translate(0, 16, &pack_off);
    ok &= ds4_expert_pack_readahead(tens[0].off, 16) && !ds4_expert_pack_readahead(0, 16);
    ds4_expert_pack_detach();
    printf(ok ? "expert pack: ok\n" : "expert pack: FAILED\n");
    return ok ? 0 : 1;
}
