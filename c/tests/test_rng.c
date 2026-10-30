#include "rng.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * RNG parity anchors, all cross-checked against the live NumPy reference
 * this session (np.random.seed(...); np.random.uniform(...)):
 *   - legacy init_genrand(0) first uniform == 0.5488135039273248
 *     (the spike's anchor and pyDeltaRCM shared_tools parity anchor)
 *   - full 6-value streams at seed 0 and seed 42, and uniform(0,9) draws
 * Full-precision expected values printed via format(v, '.17g').
 */

static int check_mt(mt_t *mt, double expected, const char *what, int idx)
{
    double v = mt_next_double(mt);
    if (v != expected) {
        printf("FAIL %s[%d]: expected %.17g, got %.17g\n", what, idx,
               expected, v);
        return 1;
    }
    return 0;
}

int main(void)
{
    int rc = 0;

    /* NumPy legacy stream, seed 0, uniform(0,1) */
    {
        static const double exp[6] = {
            0.54881350392732475, 0.71518936637241948,
            0.60276337607164387, 0.54488318299689686,
            0.42365479933890471, 0.64589411306665612};
        mt_t mt;
        mt_seed(&mt, 0);
        for (int i = 0; i < 6; i++)
            rc |= check_mt(&mt, exp[i], "seed0_unif", i);
    }

    /* NumPy legacy stream, seed 42, uniform(0,1) */
    {
        static const double exp[6] = {
            0.37454011884736249, 0.95071430640991617,
            0.73199394181140509, 0.5986584841970366,
            0.15601864044243652, 0.15599452033620265};
        mt_t mt;
        mt_seed(&mt, 42);
        for (int i = 0; i < 6; i++)
            rc |= check_mt(&mt, exp[i], "seed42_unif", i);
    }

    /* uniform(0,9): same stream, scaled by the limit */
    {
        static const double exp[6] = {
            4.9393215353459228, 6.436704297351775,
            5.4248703846447945, 4.9039486469720721,
            3.8128931940501425, 5.8130470175999047};
        mt_t mt;
        mt_seed(&mt, 0);
        for (int i = 0; i < 6; i++) {
            double v = mt_uniform_limit(&mt, 9.0);
            if (v != exp[i]) {
                printf("FAIL seed0_unif9[%d]: expected %.17g, got %.17g\n",
                       i, exp[i], v);
                rc = 1;
            }
        }
    }
    if (rc == 0) printf("test_rng: all anchors match\n");
    return rc;
}
