/* In place: rewrite the F16 hyper-connection projections of a Qwen3.8 Flash
 * Next GGUF (blk.*.hc_{attn,ffn}_{down,up}) as Q8_0.
 *
 *   cc -O2 -o qwen4_hc_q8 gguf-tools/qwen4_hc_q8.c
 *   cp -c model.gguf model-hcq8.gguf           # APFS clone, no extra space
 *   cp -c model.gguf.experts model-hcq8.gguf.experts
 *   ./qwen4_hc_q8 model-hcq8.gguf
 *
 * Each Q8_0 tensor is smaller than its F16 original, so it is written at the
 * same offset and only the tensor's type field in the header changes; the
 * tail of the old slot stays as padding, and the file size and the expert
 * bytes (so an expert pack) stay valid. Quantization follows ggml's
 * quantize_row_q8_0_ref: d = amax / 127 stored as F16, q = roundf(x * (1/d)). */
#include <math.h>
#include <regex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *f;
static uint32_t u32(void) { uint32_t v; if (fread(&v, 4, 1, f) != 1) exit(2); return v; }
static uint64_t u64(void) { uint64_t v; if (fread(&v, 8, 1, f) != 1) exit(2); return v; }
static char *gstr(void) {
    uint64_t n = u64(); char *s = malloc(n + 1);
    if (n && fread(s, n, 1, f) != 1) exit(2);
    s[n] = 0; return s;
}
static void skip(uint32_t t) {
    static const int sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t == 8) free(gstr());
    else if (t == 9) { uint32_t at = u32(); uint64_t n = u64(); for (uint64_t i = 0; i < n; i++) skip(at); }
    else fseeko(f, sz[t], SEEK_CUR);
}

typedef struct { uint64_t n, type_pos, off; char *name; } job;

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    f = fopen(argv[1], "r+b");
    if (!f) { perror(argv[1]); return 1; }
    if (u32() != 0x46554747u) { fprintf(stderr, "not GGUF\n"); return 1; }
    u32();
    uint64_t nt = u64(), nkv = u64(), align = 32;
    for (uint64_t i = 0; i < nkv; i++) {
        char *k = gstr(); uint32_t t = u32();
        if (!strcmp(k, "general.alignment")) align = u32(); else skip(t);
        free(k);
    }
    regex_t re;
    regcomp(&re, "^blk\\.[0-9]+\\.hc_(attn|ffn)_(down|up)\\.weight$", REG_EXTENDED | REG_NOSUB);
    job *jobs = calloc(nt, sizeof(job)); uint64_t nj = 0;
    for (uint64_t i = 0; i < nt; i++) {
        char *name = gstr(); uint32_t nd = u32(); uint64_t n = 1, d0 = 0;
        for (uint32_t d = 0; d < nd; d++) { uint64_t v = u64(); if (!d) d0 = v; n *= v; }
        uint64_t type_pos = (uint64_t)ftello(f); uint32_t t = u32(); uint64_t off = u64();
        if (regexec(&re, name, 0, NULL, 0) == 0) {
            if (t != 1 || d0 % 32) { fprintf(stderr, "%s: type %u dim0 %llu\n", name, t, (unsigned long long)d0); return 1; }
            jobs[nj++] = (job){n, type_pos, off, name};
        } else free(name);
    }
    const uint64_t data = ((uint64_t)ftello(f) + align - 1) / align * align;
    for (uint64_t j = 0; j < nj; j++) {
        const uint64_t n = jobs[j].n, nb = n / 32;
        _Float16 *x = malloc(n * 2);
        uint8_t *q = malloc(nb * 34);
        fseeko(f, (off_t)(data + jobs[j].off), SEEK_SET);
        if (fread(x, 2, n, f) != n) { fprintf(stderr, "read %s\n", jobs[j].name); return 1; }
        for (uint64_t b = 0; b < nb; b++) {
            float amax = 0.0f;
            for (int i = 0; i < 32; i++) { float v = fabsf((float)x[b * 32 + i]); if (v > amax) amax = v; }
            const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
            _Float16 dh = (_Float16)d;
            memcpy(q + b * 34, &dh, 2);
            for (int i = 0; i < 32; i++) q[b * 34 + 2 + i] = (uint8_t)(int8_t)roundf((float)x[b * 32 + i] * id);
        }
        fseeko(f, (off_t)(data + jobs[j].off), SEEK_SET);
        if (fwrite(q, 34, nb, f) != nb) { fprintf(stderr, "write %s\n", jobs[j].name); return 1; }
        fseeko(f, (off_t)jobs[j].type_pos, SEEK_SET);
        const uint32_t q8 = 8; fwrite(&q8, 4, 1, f);
        free(x); free(q);
    }
    if (fclose(f)) { perror("close"); return 1; }
    printf("%llu tensors converted to Q8_0\n", (unsigned long long)nj);
    return 0;
}
