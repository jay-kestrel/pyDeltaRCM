/* pyDeltaRCM COBOL kernel shim: owns the one-time GnuCOBOL runtime startup
 * (cob_init) and exposes the batched weighted-pick kernel plus the IEEE
 * helper ops the kernel uses for float arithmetic.
 *
 * Why the helpers exist: GnuCOBOL 3.2 lowers COMP COMPUTE to the GMP
 * decimal runtime, which is not bit-compatible with IEEE binary
 * arithmetic (dorado PR #32 probe: ~50% of COMP-2 multiplies differ from
 * NumPy by 1 ULP, and non-finite operands get decimalized). Each helper
 * below is exactly one C floating point op -- a single addss for the
 * float32 cumsum, one compare on the exact double widening for the
 * cutoff scan -- bit-identical to the stepwise ops the numpy/numba
 * reference performs.
 */
#include "libcob.h"
#include <stdint.h>

/* GnuCOBOL mangles PROGRAM-ID hyphens to double underscores in the
 * generated C symbols (RANDOM-PICK -> RANDOM__PICK). */
extern int RANDOM__PICK(int32_t *k, int32_t *n, float *rows, double *rns,
                        int32_t *out);

static int cobol_ready = 0;

static void ensure_cobol(void) {
    if (!cobol_ready) {
        cob_init(0, NULL);
        cobol_ready = 1;
    }
}

/* Batched weighted pick: rows is k*n float32 weights (row-major), rns is
 * k pre-drawn uniform(0, row_total) doubles, out is k picked indices
 * (0-based). Returns 0 on success, or -(row+1) when a row's total is not
 * positive (ValueError parity with the jitted reference). */
int pydeltarcm_pick_cobol(int32_t *k, int32_t *n, float *rows, double *rns,
                          int32_t *out) {
    ensure_cobol();
    return RANDOM__PICK(k, n, rows, rns, out);
}

/* IEEE-754 primitive helper: exactly one C float add, bit-identical to
 * the sequential float32 accumulation numpy's np.cumsum performs. */
void cobol_faddf(float *a, float *b, float *out) { *out = *a + *b; }

/* Exact three-way compare of a float32 cutoff (widened to double -- an
 * exact conversion, no rounding) against a double uniform, written to
 * *out: -1, 0, +1 (2 = unordered, at least one NaN). C comparisons are
 * exact on the double representation; numpy's searchsorted compares
 * float32 cumulative weights in the promoted double domain. */
void cobol_fcmp_widen(float *a, double *b, int32_t *out) {
    double da = (double)*a;
    if (da < *b) *out = -1;
    else if (da > *b) *out = 1;
    else if (da == *b) *out = 0;
    else *out = 2;
}
