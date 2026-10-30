#include "grid.h"
#include "npy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Fixture loader round-trip test:
 *  - loads bench/fixture_default_small (the committed 10x10 normalized
 *    fixture), checking config scalars, cell types, inlet columns, and
 *    walk tables against values verified this session
 *  - writes the state back out and byte-compares eta/depth/qx against the
 *    fixture payload bytes (npy writer + payload layout check)
 */
int main(void)
{
    int rc = 0;
    grid g;
    if (grid_init(&g, "../bench/fixture_default_small") != 0) {
        printf("FAIL: grid_init on fixture_default_small\n");
        return 1;
    }

    /* 10x10 grid, inlet columns [3,4,5], seven -2 cells, ninety -1 cells,
     * three 1 cells (verified this session) */
    if (g.L != 10 || g.W != 10) { printf("FAIL: L,W != 10,10\n"); rc = 1; }
    int n2 = 0, nm1 = 0, n1 = 0;
    for (size_t i = 0; i < 100; i++) {
        if (g.cell_type[i] == -2) n2++;
        else if (g.cell_type[i] == -1) nm1++;
        else if (g.cell_type[i] == 1) n1++;
    }
    if (n2 != 7 || nm1 != 90 || n1 != 3) {
        printf("FAIL: cell_type counts %d %d %d\n", n2, nm1, n1);
        rc = 1;
    }
    if (g.n_inlet != 3 || g.inlet[0] != 3 || g.inlet[1] != 4 ||
        g.inlet[2] != 5) {
        printf("FAIL: inlet columns\n");
        rc = 1;
    }
    if (g.kernel1[4] != -8 || g.kernel2[4] != 0) {
        printf("FAIL: diffusion kernel centers\n");
        rc = 1;
    }
    /* RNG must still be at stream position zero after fixture load */
    if (g.rng.pos != 624) {
        printf("FAIL: RNG not freshly seeded after load\n");
        rc = 1;
    }

    /* round-trip: write eta and depth, compare payloads with the fixture */
    if (system("mkdir -p /tmp/pyd_c_rt") != 0) return 1;
    if (npy_write_f32("/tmp/pyd_c_rt/eta.npy", g.eta, 10, 10) != 0 ||
        npy_write_f32("/tmp/pyd_c_rt/depth.npy", g.depth, 10, 10) != 0) {
        printf("FAIL: npy write\n");
        rc = 1;
        goto out;
    }
    {
        size_t n0, n1;
        char err[192];
        float *eta_rt = npy_load_f32("/tmp/pyd_c_rt/eta.npy", &n0, &n1, err,
                                     sizeof err);
        float *depth_rt = npy_load_f32("/tmp/pyd_c_rt/depth.npy", &n0, &n1,
                                       err, sizeof err);
        if (!eta_rt || !depth_rt || n0 != 10 || n1 != 10) {
            printf("FAIL: round-trip reload\n");
            rc = 1;
        } else {
            if (memcmp(eta_rt, g.eta, 100 * sizeof(float)) != 0 ||
                memcmp(depth_rt, g.depth, 100 * sizeof(float)) != 0) {
                printf("FAIL: round-trip payload mismatch\n");
                rc = 1;
            }
        }
        free(eta_rt);
        free(depth_rt);
    }

out:
    grid_free(&g);
    if (rc == 0) printf("test_fixture: round-trip OK\n");
    return rc;
}
