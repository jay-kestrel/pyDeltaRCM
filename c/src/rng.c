#include "rng.h"

#include <string.h>

/* Knuth's init_genrand, exactly as numpy's mt19937_seed for scalar seeds. */
void mt_seed(mt_t *g, uint32_t seed)
{
    g->key[0] = seed;
    for (int i = 1; i < 624; i++) {
        g->key[i] = 1812433253u * (g->key[i - 1] ^ (g->key[i - 1] >> 30)) + (uint32_t)i;
    }
    g->pos = 624;
}

static void mt_regenerate(mt_t *g)
{
    static const uint32_t MATRIX_A = 0x9908b0dfu;
    static const uint32_t UPPER = 0x80000000u;
    static const uint32_t LOWER = 0x7fffffffu;
    for (int i = 0; i < 624; i++) {
        uint32_t y = (g->key[i] & UPPER) | (g->key[(i + 1) % 624] & LOWER);
        uint32_t n = g->key[(i + 397) % 624] ^ (y >> 1);
        if (y & 1u) {
            n ^= MATRIX_A;
        }
        g->key[i] = n;
    }
    g->pos = 0;
}

uint32_t mt_next_u32(mt_t *g)
{
    if (g->pos >= 624) {
        mt_regenerate(g);
    }
    uint32_t y = g->key[g->pos++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

/* NumPy legacy double: high word first, 27 + 26 bits over 2**53. */
double mt_next_double(mt_t *g)
{
    uint32_t a = mt_next_u32(g) >> 5;
    uint32_t b = mt_next_u32(g) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

double mt_uniform_limit(mt_t *g, double limit)
{
    return mt_next_double(g) * limit;
}
