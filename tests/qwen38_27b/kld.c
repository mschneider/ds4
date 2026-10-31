/* Quality of a model against a reference model, both run by ds4 (fork tool).
 *
 *   kld --model REF.gguf  --base  BASE.bin [--tokens N] TEXT...
 *   kld --model TEST.gguf --compare BASE.bin TEXT...
 *
 * Every TEXT is tokenized raw; after a 32-token prefix its next N tokens
 * (default 1024) are teacher-forced one at a time through the ordinary
 * single-token decode path, the path the speed work changes.  --base writes
 * the full next-token log-probabilities at every position; --compare runs
 * the same positions and reports, against them:
 *
 *   KLD          KL(reference || test) over the full vocabulary (mean, p99, max)
 *   top-1        how often both pick the same most likely token
 *   ppl          perplexity of the actual next tokens, reference and test
 *
 * The reference is meant to be a high-precision file (the Q8_0 recipe, about
 * 0.001 KLD from BF16); a candidate is held to the KLD that ggml-org's
 * Q4_K_M reaches against the same reference (docs/QWEN38_27B.md). */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"

#define PREFIX 32
static const char MAGIC[8] = "Q35KLD1";

typedef struct {
    uint32_t n_vocab;
    uint32_t n_pos;
} kld_header;

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(1); }
    fseek(fp, 0, SEEK_END);
    const long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1);
    if (!s || fread(s, 1, (size_t)n, fp) != (size_t)n) { perror(path); exit(1); }
    s[n] = '\0';
    fclose(fp);
    return s;
}

/* log-softmax in double, written as float */
static void log_softmax(const float *logits, uint32_t n, float *out) {
    double mx = logits[0];
    for (uint32_t i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += exp((double)logits[i] - mx);
    const double lse = mx + log(sum);
    for (uint32_t i = 0; i < n; i++) out[i] = (float)((double)logits[i] - lse);
}

static uint32_t argmax(const float *v, uint32_t n) {
    uint32_t b = 0;
    for (uint32_t i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *model = NULL, *base = NULL, *compare = NULL;
    int n_tokens = 1024, first_text = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--base") && i + 1 < argc) base = argv[++i];
        else if (!strcmp(argv[i], "--compare") && i + 1 < argc) compare = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) n_tokens = atoi(argv[++i]);
        else { first_text = i; break; }
    }
    if (!model || (!base == !compare) || !first_text || n_tokens <= 0) {
        fprintf(stderr, "usage: %s --model GGUF (--base OUT | --compare BASE) [--tokens N] TEXT...\n", argv[0]);
        return 2;
    }

    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL };
    if (ds4_engine_open(&engine, &opt) != 0) return 1;
    const uint32_t V = (uint32_t)ds4_engine_vocab_size(engine);

    FILE *fp = fopen(base ? base : compare, base ? "wb" : "rb");
    if (!fp) { perror(base ? base : compare); return 1; }
    kld_header hdr = { V, 0 };
    if (base) {
        fwrite(MAGIC, 1, sizeof(MAGIC), fp);
        fwrite(&hdr, sizeof(hdr), 1, fp);
    } else {
        char magic[8];
        if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) || memcmp(magic, MAGIC, sizeof(magic)) ||
            fread(&hdr, sizeof(hdr), 1, fp) != 1 || hdr.n_vocab != V) {
            fprintf(stderr, "kld: %s is not a reference for this vocabulary\n", compare);
            return 1;
        }
    }

    float *logits = malloc((size_t)V * sizeof(float));
    float *lp = malloc((size_t)V * sizeof(float));
    float *ref = malloc((size_t)V * sizeof(float));
    double *klds = malloc(sizeof(double) * (size_t)n_tokens * (size_t)(argc - first_text));
    uint32_t n_pos = 0, top1 = 0;
    double nll_ref = 0.0, nll_test = 0.0, kld_sum = 0.0;

    for (int ti = first_text; ti < argc; ti++) {
        char *text = read_file(argv[ti]);
        ds4_tokens toks = {0};
        ds4_tokenize_text(engine, text, &toks);
        free(text);
        const int scored = toks.len - PREFIX - 1 < n_tokens ? toks.len - PREFIX - 1 : n_tokens;
        if (scored <= 0) { fprintf(stderr, "kld: %s is too short\n", argv[ti]); return 1; }
        ds4_session *s = NULL;
        char err[200];
        ds4_tokens prefix = {0};
        for (int i = 0; i < PREFIX; i++) ds4_tokens_push(&prefix, toks.v[i]);
        if (ds4_session_create(&s, engine, PREFIX + scored + 8) != 0 ||
            ds4_session_sync(s, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "kld: session setup failed: %s\n", err);
            return 1;
        }
        ds4_tokens_free(&prefix);
        double text_kld = 0.0;
        for (int j = 0; j < scored; j++) {
            const int next = toks.v[PREFIX + j];
            if (ds4_session_copy_logits(s, logits, (int)V) != (int)V) { fprintf(stderr, "kld: no logits\n"); return 1; }
            log_softmax(logits, V, lp);
            if (base) {
                const int32_t t = next;
                fwrite(&t, sizeof(t), 1, fp);
                fwrite(lp, sizeof(float), V, fp);
                nll_ref -= lp[next];
            } else {
                int32_t t;
                if (fread(&t, sizeof(t), 1, fp) != 1 || t != next || fread(ref, sizeof(float), V, fp) != V) {
                    fprintf(stderr, "kld: reference does not match these texts at position %u\n", n_pos);
                    return 1;
                }
                double kl = 0.0;
                for (uint32_t i = 0; i < V; i++) {
                    const double p = exp((double)ref[i]);
                    if (p > 0.0) kl += p * ((double)ref[i] - (double)lp[i]);
                }
                klds[n_pos] = kl;
                kld_sum += kl;
                text_kld += kl;
                top1 += argmax(ref, V) == argmax(lp, V);
                nll_ref -= ref[next];
                nll_test -= lp[next];
            }
            n_pos++;
            if (j + 1 < scored && ds4_session_eval(s, next, err, sizeof(err)) != 0) {
                fprintf(stderr, "kld: decode failed: %s\n", err);
                return 1;
            }
        }
        if (compare) fprintf(stderr, "kld: %-40s %5d positions  mean KLD %.5f\n", argv[ti], scored, text_kld / scored);
        ds4_session_free(s);
        ds4_tokens_free(&toks);
    }

    if (base) {
        hdr.n_pos = n_pos;
        fseek(fp, sizeof(MAGIC), SEEK_SET);
        fwrite(&hdr, sizeof(hdr), 1, fp);
        printf("reference: %u positions, ppl %.4f\n", n_pos, exp(nll_ref / n_pos));
    } else {
        if (n_pos != hdr.n_pos) { fprintf(stderr, "kld: reference has %u positions, ran %u\n", hdr.n_pos, n_pos); return 1; }
        qsort(klds, n_pos, sizeof(double), cmp_double);
        printf("positions %u  KLD mean %.5f  median %.5f  p99 %.5f  max %.4f  top-1 %.2f%%  ppl ref %.4f test %.4f\n",
               n_pos, kld_sum / n_pos, klds[n_pos / 2], klds[(size_t)(0.99 * (n_pos - 1))], klds[n_pos - 1],
               100.0 * top1 / n_pos, exp(nll_ref / n_pos), exp(nll_test / n_pos));
    }
    fclose(fp);
    ds4_engine_close(engine);
    return 0;
}
