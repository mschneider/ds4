/* Expert pack: a second copy of a model's routed experts in which each
 * expert's gate, up and down rows sit next to each other, so streaming reads
 * a missing expert from the SSD with one request instead of three.
 *
 * gguf-tools/expert_pack.py writes MODEL.gguf.experts. SSD streaming keeps
 * addressing experts by their offsets in the model file; this file only
 * redirects the reads. A read whose range lies inside one expert's slice of
 * a packed tensor goes to the pack, and reads that continue each other there
 * (gate, up, down of the same expert) are merged into one preadv.
 *
 * Included by ds4_metal.m after ds4_gpu_stream_expert_pread_task.
 * DS4_EXPERT_PACK=0 disables the pack; DS4_EXPERT_PACK=PATH names it.
 * DS4_EXPERT_PACK_VERIFY=1 compares every redirected range with the model. */

#include <sys/stat.h>
#include <sys/uio.h>

#define DS4_EXPERT_PACK_MAGIC "DS4XPAK1"
#define DS4_EXPERT_PACK_VERSION 1u
#define DS4_EXPERT_PACK_MAX_ENTRIES 4096u
#define DS4_EXPERT_PACK_CHECK_BYTES 4096u
#define DS4_EXPERT_PACK_MAX_TAILS 2u
#define DS4_EXPERT_PACK_TASK_READ 1u   /* task reads the pack */
#define DS4_EXPERT_PACK_TASK_MERGED 2u /* an earlier task reads its range */

_Static_assert(sizeof(((ds4_gpu_stream_expert_pread_task *)0)->tail_len) ==
               DS4_EXPERT_PACK_MAX_TAILS * sizeof(uint64_t),
               "pread task tails must match DS4_EXPERT_PACK_MAX_TAILS");

typedef struct {
    uint64_t model_off;    /* the tensor's first byte in the model file */
    uint64_t expert_bytes; /* one expert's slice of the tensor */
    uint64_t pack_base;    /* expert 0's copy of that slice in the pack */
    uint64_t stride;       /* bytes from one expert's record to the next */
    uint32_t n_expert;
    uint32_t reserved;
} ds4_expert_pack_entry;

typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t n_entries;
    uint64_t model_size;
    uint64_t pack_size;
    uint64_t header_bytes;
} ds4_expert_pack_header;

static int g_expert_pack_fd = -1;
static int g_expert_pack_model_fd = -1;
static dev_t g_expert_pack_model_dev;
static ino_t g_expert_pack_model_ino;
static uint32_t g_expert_pack_n_entries;
static ds4_expert_pack_entry *g_expert_pack_entries;

static void ds4_expert_pack_detach(void) {
    if (g_expert_pack_fd >= 0) close(g_expert_pack_fd);
    g_expert_pack_fd = -1;
    g_expert_pack_model_fd = -1;
    g_expert_pack_n_entries = 0;
    free(g_expert_pack_entries);
    g_expert_pack_entries = NULL;
}

static int ds4_expert_pack_pread_full(int fd, void *dst, uint64_t len, uint64_t off) {
    uint64_t pos = 0;
    while (pos < len) {
        ssize_t n;
        do {
            n = pread(fd, (uint8_t *)dst + pos, (size_t)(len - pos), (off_t)(off + pos));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return 0;
        pos += (uint64_t)n;
    }
    return 1;
}

/* The pack's bytes must equal the model's at both ends of every tensor. */
static int ds4_expert_pack_spot_check(int model_fd, int pack_fd,
                                      const ds4_expert_pack_entry *e) {
    uint8_t a[DS4_EXPERT_PACK_CHECK_BYTES], b[DS4_EXPERT_PACK_CHECK_BYTES];
    const uint64_t n = e->expert_bytes < sizeof(a) ? e->expert_bytes : sizeof(a);
    const uint64_t last = (uint64_t)e->n_expert - 1u;
    const uint64_t model[2] = {
        e->model_off,
        e->model_off + (last + 1u) * e->expert_bytes - n,
    };
    const uint64_t pack[2] = {
        e->pack_base,
        e->pack_base + last * e->stride + e->expert_bytes - n,
    };
    for (int i = 0; i < 2; i++) {
        if (!ds4_expert_pack_pread_full(model_fd, a, n, model[i]) ||
            !ds4_expert_pack_pread_full(pack_fd, b, n, pack[i]) ||
            memcmp(a, b, (size_t)n) != 0) {
            return 0;
        }
    }
    return 1;
}

static int ds4_expert_pack_validate(int model_fd, int pack_fd,
                                    const ds4_expert_pack_header *h,
                                    const ds4_expert_pack_entry *entries,
                                    const char **why) {
    struct stat ms, ps;
    if (fstat(model_fd, &ms) != 0 || fstat(pack_fd, &ps) != 0) {
        *why = "stat failed";
        return 0;
    }
    if ((uint64_t)ms.st_size != h->model_size) {
        *why = "it was written for a model of another size";
        return 0;
    }
    if ((uint64_t)ps.st_size != h->pack_size) {
        *why = "its size differs from its header (incomplete copy?)";
        return 0;
    }
    for (uint32_t i = 0; i < h->n_entries; i++) {
        const ds4_expert_pack_entry *e = &entries[i];
        if (e->n_expert == 0 || e->expert_bytes == 0 ||
            e->stride < e->expert_bytes ||
            e->expert_bytes > UINT64_MAX / e->n_expert ||
            e->model_off > h->model_size ||
            e->expert_bytes * e->n_expert > h->model_size - e->model_off ||
            (i > 0 && entries[i - 1].model_off +
                      entries[i - 1].expert_bytes * entries[i - 1].n_expert > e->model_off) ||
            e->stride > UINT64_MAX / e->n_expert ||
            e->pack_base > h->pack_size ||
            e->stride * (e->n_expert - 1u) > h->pack_size - e->pack_base ||
            e->expert_bytes > h->pack_size - e->pack_base - e->stride * (e->n_expert - 1u)) {
            *why = "its tensor table is malformed";
            return 0;
        }
        if (!ds4_expert_pack_spot_check(model_fd, pack_fd, e)) {
            *why = "its expert bytes differ from the model's";
            return 0;
        }
    }
    return 1;
}

static void ds4_expert_pack_attach(int model_fd) {
    struct stat ms;
    if (model_fd >= 0 && fstat(model_fd, &ms) == 0 &&
        model_fd == g_expert_pack_model_fd && g_expert_pack_fd >= 0 &&
        ms.st_dev == g_expert_pack_model_dev && ms.st_ino == g_expert_pack_model_ino) {
        return;
    }
    ds4_expert_pack_detach();
    if (model_fd < 0) return;

    const char *env = getenv("DS4_EXPERT_PACK");
    if (env && strcmp(env, "0") == 0) return;
    char path[PATH_MAX + 16];
    const int named = env && env[0];
    if (named) {
        snprintf(path, sizeof(path), "%s", env);
    } else {
        char model_path[PATH_MAX];
        if (fcntl(model_fd, F_GETPATH, model_path) == -1) return;
        snprintf(path, sizeof(path), "%s.experts", model_path);
    }

    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (named) fprintf(stderr, "ds4: expert pack %s: %s\n", path, strerror(errno));
        return;
    }
    ds4_expert_pack_header h;
    ds4_expert_pack_entry *entries = NULL;
    const char *why = "its header is unreadable";
    int ok = ds4_expert_pack_pread_full(fd, &h, sizeof(h), 0) &&
             memcmp(h.magic, DS4_EXPERT_PACK_MAGIC, 8) == 0 &&
             h.version == DS4_EXPERT_PACK_VERSION &&
             h.n_entries != 0 && h.n_entries <= DS4_EXPERT_PACK_MAX_ENTRIES &&
             h.header_bytes >= sizeof(h) + h.n_entries * sizeof(*entries);
    if (ok) {
        entries = calloc(h.n_entries, sizeof(*entries));
        ok = entries &&
             ds4_expert_pack_pread_full(fd, entries, h.n_entries * sizeof(*entries), sizeof(h)) &&
             ds4_expert_pack_validate(model_fd, fd, &h, entries, &why);
    }
    if (!ok) {
        fprintf(stderr, "ds4: warning: ignoring expert pack %s: %s\n", path, why);
        free(entries);
        close(fd);
        return;
    }

    g_expert_pack_fd = fd;
    g_expert_pack_model_fd = model_fd;
    g_expert_pack_model_dev = ms.st_dev;
    g_expert_pack_model_ino = ms.st_ino;
    g_expert_pack_n_entries = h.n_entries;
    g_expert_pack_entries = entries;
    fprintf(stderr, "ds4: expert pack %s: %u tensors, %.2f GiB, one read per expert\n",
            path, h.n_entries, (double)h.pack_size / (1024.0 * 1024.0 * 1024.0));
}

/* Model-file range -> pack offset, when the range lies in one expert slice. */
static int ds4_expert_pack_translate(uint64_t off, uint64_t len, uint64_t *pack_off) {
    if (g_expert_pack_fd < 0 || len == 0) return 0;
    uint32_t lo = 0, hi = g_expert_pack_n_entries;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (g_expert_pack_entries[mid].model_off <= off) lo = mid + 1u;
        else hi = mid;
    }
    if (lo == 0) return 0;
    const ds4_expert_pack_entry *e = &g_expert_pack_entries[lo - 1u];
    const uint64_t rel = off - e->model_off;
    const uint64_t expert = rel / e->expert_bytes;
    const uint64_t inner = rel % e->expert_bytes;
    if (expert >= e->n_expert || len > e->expert_bytes - inner) return 0;
    *pack_off = e->pack_base + expert * e->stride + inner;
    return 1;
}

static int ds4_expert_pack_readahead(uint64_t off, uint64_t len) {
    uint64_t pack_off;
    if (!ds4_expert_pack_translate(off, len, &pack_off)) return 0;
#if defined(F_RDADVISE)
    if (len <= (uint64_t)INT_MAX) {
        struct radvisory ra = { .ra_offset = (off_t)pack_off, .ra_count = (int)len };
        (void)fcntl(g_expert_pack_fd, F_RDADVISE, &ra);
    }
#endif
    return 1;
}

static uint64_t ds4_expert_pack_task_bytes(const ds4_gpu_stream_expert_pread_task *t) {
    uint64_t n = t->len;
    for (uint32_t i = 0; i < t->pack_tails; i++) n += t->tail_len[i];
    return n;
}

/* Point the tasks at the pack and merge each run of tasks whose pack ranges
 * continue one another into its first task. Merged tasks are left empty. */
static void ds4_expert_pack_coalesce(ds4_gpu_stream_expert_pread_task *tasks,
                                     uint32_t n_tasks) {
    if (g_expert_pack_fd < 0) return;
    static int verify = -1;
    if (verify < 0) verify = getenv("DS4_EXPERT_PACK_VERIFY") != NULL;
    for (uint32_t i = 0; i < n_tasks; i++) {
        ds4_gpu_stream_expert_pread_task *t = &tasks[i];
        uint64_t pack_off;
        if (t->pack == 0 && ds4_expert_pack_translate(t->offset, t->len, &pack_off)) {
            if (verify) {
                /* Debug: the redirected range must hold the model's bytes. */
                uint8_t *a = malloc((size_t)t->len), *b = malloc((size_t)t->len);
                if (!a || !b ||
                    !ds4_expert_pack_pread_full(g_expert_pack_model_fd, a, t->len, t->offset) ||
                    !ds4_expert_pack_pread_full(g_expert_pack_fd, b, t->len, pack_off) ||
                    memcmp(a, b, (size_t)t->len) != 0) {
                    fprintf(stderr, "ds4: expert pack verify failed model_off=%llu len=%llu\n",
                            (unsigned long long)t->offset, (unsigned long long)t->len);
                    abort();
                }
                free(a);
                free(b);
            }
            t->pack = DS4_EXPERT_PACK_TASK_READ;
            t->offset = pack_off;
        }
    }
    ds4_gpu_stream_expert_pread_task *head = NULL;
    for (uint32_t i = 0; i < n_tasks; i++) {
        ds4_gpu_stream_expert_pread_task *t = &tasks[i];
        if (t->pack != DS4_EXPERT_PACK_TASK_READ) {
            head = NULL;
            continue;
        }
        if (head && head->pack_tails < DS4_EXPERT_PACK_MAX_TAILS &&
            head->offset + ds4_expert_pack_task_bytes(head) == t->offset &&
            t->pack_tails == 0) {
            head->tail_dst[head->pack_tails] = t->dst;
            head->tail_len[head->pack_tails] = t->len;
            head->pack_tails++;
            t->pack = DS4_EXPERT_PACK_TASK_MERGED;
            continue;
        }
        head = t;
    }
}

/* One task's read from the pack: its own range and its merged tails. */
static int ds4_expert_pack_read_task(ds4_gpu_stream_expert_pread_task *t) {
    struct iovec iov[1 + DS4_EXPERT_PACK_MAX_TAILS];
    int n_iov = 0;
    iov[n_iov++] = (struct iovec){ .iov_base = t->dst, .iov_len = (size_t)t->len };
    for (uint32_t i = 0; i < t->pack_tails; i++) {
        iov[n_iov++] = (struct iovec){ .iov_base = t->tail_dst[i], .iov_len = (size_t)t->tail_len[i] };
    }
    const uint64_t total = ds4_expert_pack_task_bytes(t);
    const double t0 = ds4_gpu_now_ms();
    uint64_t done = 0;
    int first = 0;
    while (done < total) {
        ssize_t n;
        do {
            n = preadv(g_expert_pack_fd, iov + first, n_iov - first, (off_t)(t->offset + done));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) break;
        done += (uint64_t)n;
        /* Skip the vectors a short read completed; trim the one it split. */
        size_t left = (size_t)n;
        while (first < n_iov && left >= iov[first].iov_len) left -= iov[first++].iov_len;
        if (first < n_iov && left) {
            iov[first].iov_base = (uint8_t *)iov[first].iov_base + left;
            iov[first].iov_len -= left;
        }
    }
    t->read_bytes = done;
    t->ms = ds4_gpu_now_ms() - t0;
    if (done != total) {
        fprintf(stderr, "ds4: expert pack read failed offset=%.2f GiB len=%.2f MiB read=%.2f MiB\n",
                (double)t->offset / (1024.0 * 1024.0 * 1024.0),
                (double)total / (1024.0 * 1024.0),
                (double)done / (1024.0 * 1024.0));
        return 0;
    }
    return 1;
}

static int ds4_expert_pack_run_task(ds4_gpu_stream_expert_pread_task *t) {
    if (t->pack == DS4_EXPERT_PACK_TASK_MERGED) {
        t->read_bytes = 0;
        t->ms = 0.0;
        return 1;
    }
    return ds4_expert_pack_read_task(t);
}
