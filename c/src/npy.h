#ifndef PYD_NPY_H
#define PYD_NPY_H

#include <stddef.h>
#include <stdint.h>

/*
 * Minimal .npy reader/writer (v1.0 header format, C-order payloads).
 *
 * Covers exactly the dtypes the fixtures and outputs use: float32/float64/
 * int64/int32/uint8. Reader rejects Fortran order (fixtures are C order).
 * Writer emits numpy-readable v1.0 headers padded to a 64-byte boundary.
 */
typedef enum { NPY_F32, NPY_F64, NPY_I64, NPY_I32, NPY_U8 } npy_dtype;

typedef struct {
    npy_dtype dtype;
    size_t    ndim;        /* 1 or 2 */
    size_t    shape[2];    /* [0] rows (or count), [1] cols (0 for 1D) */
    void     *data;        /* rows*cols elements */
} npy_t;

/* Read a .npy file. Returns 0 on success; on failure returns -1 and the
 * reason is written to err (may be NULL). Caller frees with npy_free. */
int npy_read(const char *path, npy_t *out, char *err, size_t errlen);
void npy_free(npy_t *n);

/* Typed loaders: read any supported dtype and cast to the target type.
 * Returns NULL on failure (reason via err). Caller frees. */
float  *npy_load_f32(const char *path, size_t *n0, size_t *n1, char *err, size_t errlen);
double *npy_load_f64(const char *path, size_t *n0, size_t *n1, char *err, size_t errlen);
int64_t *npy_load_i64(const char *path, size_t *n0, size_t *n1, char *err, size_t errlen);
int32_t *npy_load_i32(const char *path, size_t *n0, size_t *n1, char *err, size_t errlen);

/* Write a 2-D (or 1-D with cols==0) array as .npy. */
int npy_write_f32(const char *path, const float *a, size_t n0, size_t n1);
int npy_write_f64(const char *path, const double *a, size_t n0, size_t n1);
int npy_write_i64(const char *path, const int64_t *a, size_t n0, size_t n1);

#endif
