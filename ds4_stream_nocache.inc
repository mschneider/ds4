/* Uncached reads for SSD expert streaming (fork).
 *
 * Streamed experts are read with pread into the expert cache's own buffers,
 * so the OS file cache only holds a second copy of them. That copy competes
 * with the expert cache and the rest of the system for memory, and it makes a
 * cache miss cost anywhere from a RAM copy to an SSD read depending on what
 * the OS happened to keep. With DS4_STREAM_NOCACHE=1 the streaming reads go
 * through a second descriptor of the model file with F_NOCACHE set (the
 * model's own descriptor, which ds4.c maps, is left alone), the expert pack
 * descriptor gets F_NOCACHE too, and the F_RDADVISE readahead before each
 * read is skipped because it would fill the file cache again.
 *
 * Included by ds4_metal.m after g_model_fd. */

#include <sys/stat.h>

static int g_stream_nocache = -1;
static int g_stream_nocache_fd = -1;
static dev_t g_stream_nocache_dev;
static ino_t g_stream_nocache_ino;

static int ds4_stream_nocache_enabled(void) {
    if (g_stream_nocache < 0) {
        const char *env = getenv("DS4_STREAM_NOCACHE");
        g_stream_nocache = env && strcmp(env, "1") == 0;
    }
    return g_stream_nocache;
}

static void ds4_stream_nocache_close(void) {
    if (g_stream_nocache_fd >= 0) close(g_stream_nocache_fd);
    g_stream_nocache_fd = -1;
}

/* Returns the descriptor the streaming reads should use for model_fd. */
static int ds4_stream_nocache_fd_for(int model_fd) {
    if (model_fd < 0 || !ds4_stream_nocache_enabled()) return model_fd;
    struct stat ms;
    if (fstat(model_fd, &ms) != 0) return model_fd;
    if (g_stream_nocache_fd >= 0 &&
        ms.st_dev == g_stream_nocache_dev && ms.st_ino == g_stream_nocache_ino) {
        return g_stream_nocache_fd;
    }
    ds4_stream_nocache_close();
    char path[PATH_MAX];
    struct stat ns;
    int fd = -1;
    if (fcntl(model_fd, F_GETPATH, path) != -1) fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0 && (fstat(fd, &ns) != 0 || ns.st_dev != ms.st_dev || ns.st_ino != ms.st_ino ||
                    fcntl(fd, F_NOCACHE, 1) == -1)) {
        close(fd);
        fd = -1;
    }
    if (fd < 0) {
        fprintf(stderr, "ds4: warning: DS4_STREAM_NOCACHE: cannot reopen the model; reads stay cached\n");
        return model_fd;
    }
    g_stream_nocache_fd = fd;
    g_stream_nocache_dev = ms.st_dev;
    g_stream_nocache_ino = ms.st_ino;
    return fd;
}
