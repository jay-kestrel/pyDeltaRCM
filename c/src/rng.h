#ifndef PYD_MT19937_H
#define PYD_MT19937_H

#include <stdint.h>
#include <stddef.h>

/*
 * MT19937 with NumPy legacy RandomState semantics.
 *
 * Replicates np.random.seed(int) / np.random.random_sample() bit-for-bit:
 *   - scalar seeds < 2**32 initialize via Knuth's init_genrand
 *   - doubles are the legacy two-word conversion
 *     ((a >> 5) * 2**26 + (b >> 6)) / 2**53 with the high word drawn first
 *
 * The RNG-spike findings (task todo_4jSfu2tQ) verified that Numba's njit
 * np.random stream is bit-identical to this legacy stream for the router
 * draw patterns, so this generator is the single parity stream for the port.
 */
typedef struct {
    uint32_t key[624];
    size_t   pos;
} mt_t;

void     mt_seed(mt_t *g, uint32_t seed);
uint32_t mt_next_u32(mt_t *g);
double   mt_next_double(mt_t *g);

/* np.random.uniform(0, limit): 0 + (limit - 0) * random_sample() */
double   mt_uniform_limit(mt_t *g, double limit);

#endif
