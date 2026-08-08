/* Thin libc wrappers under nomos_ prefix so the Mojo kernel can call
 * file-I/O syscalls without colliding with the Mojo stdlib's internal
 * external_call declarations of the same libc symbols (open/read/close/
 * lseek/pread/mmap/munmap). Each wrapper takes the same arguments as
 * the underlying libc function in the same order. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int nomos_open(const char *path, int flags, int mode) {
    /* mode is only honored when O_CREAT or O_TMPFILE is in flags;
     * ignored otherwise. Always passing it (rather than using a 2-arg
     * open) keeps glibc happy — recent glibc aborts the process if
     * O_CREAT is requested without an explicit mode argument. */
    return open(path, flags, mode);
}

int nomos_close(int fd) {
    return close(fd);
}

ssize_t nomos_read(int fd, void *buf, size_t count) {
    return read(fd, buf, count);
}

ssize_t nomos_pread(int fd, void *buf, size_t count, off_t offset) {
    return pread(fd, buf, count, offset);
}

off_t nomos_lseek(int fd, off_t offset, int whence) {
    return lseek(fd, offset, whence);
}

void *nomos_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    return mmap(addr, length, prot, flags, fd, offset);
}

int nomos_munmap(void *addr, size_t length) {
    return munmap(addr, length);
}

int nomos_mkdir(const char *path, mode_t mode) {
    return mkdir(path, mode);
}

ssize_t nomos_send(int sockfd, const void *buf, size_t len, int flags) {
    return send(sockfd, buf, len, flags);
}

ssize_t nomos_recv(int sockfd, void *buf, size_t len, int flags) {
    return recv(sockfd, buf, len, flags);
}

FILE *nomos_fopen(const char *path, const char *mode) {
    return fopen(path, mode);
}

FILE *nomos_fdopen(int fd, const char *mode) {
    return fdopen(fd, mode);
}

int nomos_fclose(FILE *fp) {
    return fclose(fp);
}

int nomos_fflush(FILE *fp) {
    return fflush(fp);
}

int nomos_fsync(int fd) {
    return fsync(fd);
}

size_t nomos_fwrite(const void *ptr, size_t size, size_t nmemb, FILE *fp) {
    return fwrite(ptr, size, nmemb, fp);
}

size_t nomos_fread(void *ptr, size_t size, size_t nmemb, FILE *fp) {
    return fread(ptr, size, nmemb, fp);
}

/* Cached environment-flag reader. The Mojo decode dispatch reads several
 * immutable NOMOS_* toggles once *per GEMV* — ~1000 getenv calls/token via
 * _read_env_bytes, each doing a libc environ scan + 2 heap allocations.
 * These flags never change after process launch, so cache them: getenv once
 * per id, return the cached tri-state thereafter. `id` MUST stay in sync with
 * the Mojo ENV_* aliases in lib/engine_init.mojo and the order below.
 * Returns: -1 unset, 0 = "0", 1 = "1", 2 = other set value. */
static const char *const NOMOS_FLAG_NAMES[] = {
    "NOMOS_NVFP4_V3",        /* 0 */
    "NOMOS_NVFP4_V3_TREE",   /* 1 */
    "NOMOS_NVFP4_FUSED",     /* 2 */
    "NOMOS_Q4_FUSED",        /* 3 */
    "NOMOS_NVFP4_GEMV_V2",   /* 4 */
    "NOMOS_Q4_DP4A",         /* 5 — Q4 decode acts: =1 q8 dp4a (prod), else fp32 occ4 */
    "NOMOS_STRICT_Q4",       /* 6 — precision-law guard: =1 record any non-Q4/Q8 path reached */
};
#define NOMOS_NFLAGS ((int)(sizeof(NOMOS_FLAG_NAMES) / sizeof(NOMOS_FLAG_NAMES[0])))
/* -2 = uncomputed sentinel; keep the initializer length == NOMOS_NFLAGS. */
static int nomos_flag_cache[NOMOS_NFLAGS] = {-2, -2, -2, -2, -2, -2, -2};

int nomos_env_flag_cached(int id) {
    if (id < 0 || id >= NOMOS_NFLAGS) return -1;
    int c = nomos_flag_cache[id];
    if (c != -2) return c;
    const char *v = getenv(NOMOS_FLAG_NAMES[id]);
    if (v == NULL)                     c = -1;
    else if (v[0] == '0' && v[1] == 0) c = 0;
    else if (v[0] == '1' && v[1] == 0) c = 1;
    else                               c = 2;
    nomos_flag_cache[id] = c;
    return c;
}

/* ── Precision-law guard tripwire ──────────────────────────────────────────
 * THE LAW: weights Q4, acts Q8, KV INT4; fp32 only as a transient accumulator
 * register; bf16/fp16 NEVER; no fp32 matmul on a weight/activation. Each
 * forbidden precision DISPATCH calls nomos_flag_violation(code) when
 * NOMOS_STRICT_Q4=1 — sets bit `code` in a mask + counts hits, so one run
 * ENUMERATES every reachable violation at once. Codes mirror VIOL_* in
 * lib/engine_init.mojo. count()==0 under the Q4 production config == green. */
static long nomos_viol_count = 0;
static long nomos_viol_mask  = 0;

int nomos_flag_violation(int code) {
    nomos_viol_count++;
    if (code >= 0 && code < 63) nomos_viol_mask |= (1L << code);
    return 0;
}
long nomos_violation_count(void) { return nomos_viol_count; }
long nomos_violation_mask(void)  { return nomos_viol_mask; }
int  nomos_reset_violations(void) { nomos_viol_count = 0; nomos_viol_mask = 0; return 0; }
