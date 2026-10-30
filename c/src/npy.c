#include "npy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t dtype_size(npy_dtype dt)
{
    switch (dt) {
    case NPY_F32: return 4;
    case NPY_F64: return 8;
    case NPY_I64: return 8;
    case NPY_I32: return 4;
    case NPY_U8:  return 1;
    }
    return 0;
}

static npy_dtype descr_to_dtype(const char *d)
{
    if (!strcmp(d, "<f4") || !strcmp(d, "|f4")) return NPY_F32;
    if (!strcmp(d, "<f8") || !strcmp(d, "|f8")) return NPY_F64;
    if (!strcmp(d, "<i8") || !strcmp(d, "|i8")) return NPY_I64;
    if (!strcmp(d, "<i4") || !strcmp(d, "|i4")) return NPY_I32;
    if (!strcmp(d, "|b1") || !strcmp(d, "|u1")) return NPY_U8;
    return (npy_dtype)-1;
}

/* Extract "key": <value> from a numpy header dict. Returns the value token
 * start, or NULL. Handles both quoted and bare values. */
static const char *hdr_value(const char *hdr, const char *key)
{
    char pat[64];
    snprintf(pat, sizeof pat, "'%s'", key);
    const char *p = strstr(hdr, pat);
    if (!p) return NULL;
    p = strchr(p + strlen(pat), ':');
    if (!p) return NULL;
    p++;
    while (*p == ' ') p++;
    return p;
}

int npy_read(const char *path, npy_t *out, char *err, size_t errlen)
{
    memset(out, 0, sizeof *out);
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (err) snprintf(err, errlen, "cannot open %s", path);
        return -1;
    }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        if (err) snprintf(err, errlen, "%s: bad magic", path);
        fclose(f);
        return -1;
    }
    uint32_t hlen;
    if (magic[6] == 1) {
        unsigned char h2[2];
        if (fread(h2, 1, 2, f) != 2) { fclose(f); return -1; }
        hlen = (uint32_t)h2[0] | ((uint32_t)h2[1] << 8);
    } else {
        unsigned char h4[4];
        if (fread(h4, 1, 4, f) != 4) { fclose(f); return -1; }
        hlen = (uint32_t)h4[0] | ((uint32_t)h4[1] << 8) |
               ((uint32_t)h4[2] << 16) | ((uint32_t)h4[3] << 24);
    }
    char *hdr = malloc(hlen + 1);
    if (!hdr || fread(hdr, 1, hlen, f) != hlen) {
        free(hdr);
        fclose(f);
        if (err) snprintf(err, errlen, "%s: truncated header", path);
        return -1;
    }
    hdr[hlen] = '\0';

    if (strstr(hdr, "'fortran_order': True")) {
        free(hdr);
        fclose(f);
        if (err) snprintf(err, errlen, "%s: Fortran order unsupported", path);
        return -1;
    }
    const char *dv = hdr_value(hdr, "descr");
    char descr[8] = {0};
    if (!dv || *dv != '\'') { free(hdr); fclose(f); return -1; }
    dv++;
    size_t di = 0;
    while (*dv && *dv != '\'' && di + 1 < sizeof descr) descr[di++] = *dv++;
    descr[di] = '\0';
    out->dtype = descr_to_dtype(descr);
    if ((int)out->dtype < 0 || dtype_size(out->dtype) == 0) {
        free(hdr); fclose(f);
        if (err) snprintf(err, errlen, "%s: unsupported descr %s", path, descr);
        return -1;
    }
    const char *sv = hdr_value(hdr, "shape");
    if (!sv || *sv != '(') { free(hdr); fclose(f); return -1; }
    sv++;
    size_t dims[2] = {0, 0};
    int nd = 0;
    while (*sv && *sv != ')' && nd < 2) {
        char *endp;
        long v = strtol(sv, &endp, 10);
        if (endp == sv) break;
        dims[nd++] = (size_t)v;
        sv = endp;
        while (*sv == ' ' || *sv == ',') sv++;
    }
    free(hdr);
    out->ndim = (size_t)nd;
    out->shape[0] = dims[0];
    out->shape[1] = nd == 2 ? dims[1] : 0;
    size_t count = dims[0] * (nd == 2 ? dims[1] : 1);
    size_t total = count * dtype_size(out->dtype);
    out->data = malloc(total ? total : 1);
    if (!out->data || fread(out->data, 1, total, f) != total) {
        npy_free(out);
        fclose(f);
        if (err) snprintf(err, errlen, "%s: truncated payload", path);
        return -1;
    }
    fclose(f);
    return 0;
}

void npy_free(npy_t *n)
{
    free(n->data);
    n->data = NULL;
}

static const void *as_cast(const npy_t *n, size_t idx)
{
    static double df;
    static float ff;
    static int64_t i64;
    static int32_t i32;
    static uint8_t u8;
    void *p = (char *)n->data + idx * dtype_size(n->dtype);
    switch (n->dtype) {
    case NPY_F32: ff = *(float *)p; return &ff;
    case NPY_F64: df = *(double *)p; return &df;
    case NPY_I64: i64 = *(int64_t *)p; return &i64;
    case NPY_I32: i32 = *(int32_t *)p; return &i32;
    case NPY_U8:  u8 = *(uint8_t *)p; return &u8;
    }
    return NULL;
}

#define DEFINE_TYPED_LOAD(CType, SUFFIX)                                      \
    CType *npy_load_##SUFFIX(const char *path, size_t *n0, size_t *n1,        \
                             char *err, size_t errlen)                        \
    {                                                                          \
        npy_t n;                                                               \
        if (npy_read(path, &n, err, errlen) != 0) return NULL;                 \
        size_t count = n.shape[0] * (n.ndim == 2 ? n.shape[1] : 1);            \
        CType *out = malloc((count ? count : 1) * sizeof(CType));              \
        if (!out) { npy_free(&n); return NULL; }                               \
        for (size_t i = 0; i < count; i++) {                                   \
            const void *v = as_cast(&n, i);                                    \
            if (n.dtype == NPY_F32) out[i] = (CType)*(const float *)v;         \
            else if (n.dtype == NPY_F64) out[i] = (CType)*(const double *)v;   \
            else if (n.dtype == NPY_I64) out[i] = (CType)*(const int64_t *)v;  \
            else if (n.dtype == NPY_I32) out[i] = (CType)*(const int32_t *)v;  \
            else out[i] = (CType)*(const uint8_t *)v;                          \
        }                                                                      \
        if (n0) *n0 = n.shape[0];                                              \
        if (n1) *n1 = n.ndim == 2 ? n.shape[1] : 0;                            \
        npy_free(&n);                                                          \
        return out;                                                            \
    }

DEFINE_TYPED_LOAD(float, f32)
DEFINE_TYPED_LOAD(double, f64)
DEFINE_TYPED_LOAD(int64_t, i64)
DEFINE_TYPED_LOAD(int32_t, i32)

static int npy_write_any(const char *path, const char *descr, size_t elemsize,
                         const void *a, size_t n0, size_t n1)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    char dict[160];
    if (n1)
        snprintf(dict, sizeof dict,
                 "{'descr': '%s', 'fortran_order': False, 'shape': (%zu, %zu), }",
                 descr, n0, n1);
    else
        snprintf(dict, sizeof dict,
                 "{'descr': '%s', 'fortran_order': False, 'shape': (%zu,), }",
                 descr, n0);
    size_t dlen = strlen(dict);
    /* 10-byte magic+version+header-length prefix; pad to 64-byte boundary */
    size_t hlen = dlen + 1;
    while ((10 + hlen) % 64 != 0) hlen++;
    if (hlen > 65535) { fclose(f); return -1; }
    unsigned char pre[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                             (unsigned char)(hlen & 0xff),
                             (unsigned char)((hlen >> 8) & 0xff)};
    fwrite(pre, 1, 10, f);
    fwrite(dict, 1, dlen, f);
    for (size_t i = dlen; i < hlen - 1; i++) fputc(' ', f);
    fputc('\n', f);
    size_t count = n0 * (n1 ? n1 : 1);
    fwrite(a, elemsize, count, f);
    int werr = ferror(f);
    fclose(f);
    return werr ? -1 : 0;
}

int npy_write_f32(const char *path, const float *a, size_t n0, size_t n1)
{
    return npy_write_any(path, "<f4", 4, a, n0, n1);
}
int npy_write_f64(const char *path, const double *a, size_t n0, size_t n1)
{
    return npy_write_any(path, "<f8", 8, a, n0, n1);
}
int npy_write_i64(const char *path, const int64_t *a, size_t n0, size_t n1)
{
    return npy_write_any(path, "<i8", 8, a, n0, n1);
}
