#include "grid.h"
#include "model.h"
#include "npy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * pydeltarcm_c — standalone pyDeltaRCM port driver.
 *
 *   pydeltarcm_c --config bench/fixture_default/config.json \
 *                 --updates 10 --out out/c/
 *
 * Loads the normalized fixture (config.json + state/*.npy), runs N updates
 * (compute clock excludes all I/O), writes the state arrays as .npy plus a
 * results.json manifest under --out.
 */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void usage(void)
{
    fprintf(stderr,
            "usage: pydeltarcm_c --config <fixture_dir> [--updates N] "
            "--out <dir>\n");
    exit(2);
}

static int write_outputs(const grid *g, const char *outdir)
{
    int L = g->L, W = g->W;
    struct { const char *name; const float *a; } f32s[] = {
        {"eta.npy", g->eta},         {"depth.npy", g->depth},
        {"stage.npy", g->stage},     {"Hnew.npy", g->Hnew},
        {"qw.npy", g->qw},           {"qx.npy", g->qx},
        {"qy.npy", g->qy},           {"qwn.npy", g->qwn},
        {"uw.npy", g->uw},           {"ux.npy", g->ux},
        {"uy.npy", g->uy},           {"qs.npy", g->qs},
        {"sand_frac.npy", g->sand_frac},
        {"active_layer.npy", g->active_layer},
        {"Vp_dep_sand.npy", g->Vp_dep_sand},
        {"Vp_dep_mud.npy", g->Vp_dep_mud},
    };
    for (size_t i = 0; i < sizeof f32s / sizeof f32s[0]; i++) {
        char *p = malloc(strlen(outdir) + strlen(f32s[i].name) + 2);
        sprintf(p, "%s/%s", outdir, f32s[i].name);
        int rc = npy_write_f32(p, f32s[i].a, (size_t)L, (size_t)W);
        free(p);
        if (rc != 0) return -1;
    }
    {
        char *p = malloc(strlen(outdir) + strlen("cell_type.npy") + 2);
        sprintf(p, "%s/cell_type.npy", outdir);
        int rc = npy_write_i64(p, g->cell_type, (size_t)L, (size_t)W);
        free(p);
        if (rc != 0) return -1;
    }
    {
        char *p = malloc(strlen(outdir) + strlen("water_weights.npy") + 2);
        sprintf(p, "%s/water_weights.npy", outdir);
        int rc = npy_write_f64(p, g->water_weights, (size_t)L * W, 9);
        free(p);
        if (rc != 0) return -1;
    }
    return 0;
}

static int write_results(const grid *g, const char *fixture, int updates,
                         double compute_seconds, const char *outdir)
{
    char *p = malloc(strlen(outdir) + strlen("results.json") + 2);
    sprintf(p, "%s/results.json", outdir);
    FILE *f = fopen(p, "w");
    free(p);
    if (!f) return -1;
    fprintf(f,
            "{\n"
            "  \"model\": \"pydeltarcm_c\",\n"
            "  \"fixture\": \"%s\",\n"
            "  \"updates\": %d,\n"
            "  \"L\": %d,\n"
            "  \"W\": %d,\n"
            "  \"seed\": %llu,\n"
            "  \"Np_water\": %d,\n"
            "  \"Np_sed\": %d,\n"
            "  \"itermax\": %d,\n"
            "  \"compute_seconds\": %.9f,\n"
            "  \"final_time\": %.17g,\n"
            "  \"time_iter\": %ld\n"
            "}\n",
            fixture, updates, g->L, g->W, (unsigned long long)g->seed,
            g->Np_water, g->Np_sed, g->itermax, compute_seconds, g->time,
            g->time_iter);
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    const char *fixture = NULL, *outdir = NULL;
    int updates = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) {
            fixture = argv[++i];
        } else if (!strcmp(argv[i], "--updates") && i + 1 < argc) {
            updates = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            outdir = argv[++i];
        } else {
            usage();
        }
    }
    if (!fixture || !outdir) usage();

    grid g;
    if (grid_init(&g, fixture) != 0) return 1;

    double t0 = now_sec();
    int urc = updates > 0 ? pyd_update(&g, updates) : 0;
    double t1 = now_sec();
    if (urc != 0) {
        grid_free(&g);
        return 3;
    }

    if (write_outputs(&g, outdir) != 0) {
        fprintf(stderr, "pydeltarcm_c: failed writing outputs to %s\n", outdir);
        grid_free(&g);
        return 1;
    }
    if (write_results(&g, fixture, updates, t1 - t0, outdir) != 0) {
        fprintf(stderr, "pydeltarcm_c: failed writing results.json\n");
        grid_free(&g);
        return 1;
    }
    printf("pydeltarcm_c: %dx%d, %d updates, %.6fs compute\n", g.L, g.W,
           updates, t1 - t0);
    grid_free(&g);
    return 0;
}
