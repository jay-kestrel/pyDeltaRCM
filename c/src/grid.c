#include "grid.h"

#include "json.h"
#include "npy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *join_path(const char *dir, const char *leaf)
{
    size_t n = strlen(dir) + strlen(leaf) + 2;
    char *s = malloc(n);
    snprintf(s, n, "%s/%s", dir, leaf);
    return s;
}

/* Load a float32 L*W state array. */
static float *load_state_f32(const char *dir, const char *name, size_t count)
{
    char *p = join_path(dir, name);
    size_t n0, n1;
    char err[192];
    float *a = npy_load_f32(p, &n0, &n1, err, sizeof err);
    if (!a) {
        fprintf(stderr, "pydeltarcm_c: %s\n", err);
        free(p);
        exit(1);
    }
    free(p);
    if (n0 * (n1 ? n1 : 1) != count) {
        fprintf(stderr, "pydeltarcm_c: %s has wrong size\n", name);
        free(a);
        exit(1);
    }
    return a;
}

static int64_t *load_state_i64(const char *dir, const char *name, size_t count)
{
    char *p = join_path(dir, name);
    size_t n0, n1;
    char err[192];
    int64_t *a = npy_load_i64(p, &n0, &n1, err, sizeof err);
    if (!a) {
        fprintf(stderr, "pydeltarcm_c: %s\n", err);
        free(p);
        exit(1);
    }
    free(p);
    if (n0 * (n1 ? n1 : 1) != count) {
        fprintf(stderr, "pydeltarcm_c: %s has wrong size\n", name);
        free(a);
        exit(1);
    }
    return a;
}

/* Load a float32 9-entry table. */
static void load_table_f32(const char *dir, const char *name, float out[9])
{
    size_t n0, n1;
    char err[192];
    char *p = join_path(dir, name);
    float *a = npy_load_f32(p, &n0, &n1, err, sizeof err);
    free(p);
    if (!a || n0 * (n1 ? n1 : 1) != 9) {
        fprintf(stderr, "pydeltarcm_c: %s must have 9 entries\n", name);
        free(a);
        exit(1);
    }
    memcpy(out, a, 9 * sizeof(float));
    free(a);
}

static void load_table_i64(const char *dir, const char *name, int64_t out[9])
{
    size_t n0, n1;
    char err[192];
    char *p = join_path(dir, name);
    int64_t *a = npy_load_i64(p, &n0, &n1, err, sizeof err);
    free(p);
    if (!a || n0 * (n1 ? n1 : 1) != 9) {
        fprintf(stderr, "pydeltarcm_c: %s must have 9 entries\n", name);
        free(a);
        exit(1);
    }
    memcpy(out, a, 9 * sizeof(int64_t));
    free(a);
}

void grid_pad_f32(const float *src, float *dst, int L, int W)
{
    /* np.pad(arr, 1, 'edge'): replicate the border cells outward */
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < W; j++) {
            dst[(size_t)(i + 1) * (W + 2) + (j + 1)] = src[(size_t)i * W + j];
        }
    }
    for (int i = 0; i < L; i++) {
        dst[(size_t)(i + 1) * (W + 2) + 0] = src[(size_t)i * W];            /* left */
        dst[(size_t)(i + 1) * (W + 2) + (W + 1)] = src[(size_t)i * W + W - 1]; /* right */
    }
    for (int j = 0; j < W; j++) {
        dst[0 * (W + 2) + (j + 1)] = src[0 * W + j];                        /* top */
        dst[(size_t)(L + 1) * (W + 2) + (j + 1)] = src[(size_t)(L - 1) * W + j]; /* bottom */
    }
    dst[0] = src[0];                                        /* ul */
    dst[W + 1] = src[W - 1];                                /* ur */
    dst[(size_t)(L + 1) * (W + 2)] = src[(size_t)(L - 1) * W];          /* ll */
    dst[(size_t)(L + 1) * (W + 2) + (W + 1)] = src[(size_t)(L - 1) * W + W - 1]; /* lr */
}

void grid_pad_i64(const int64_t *src, int64_t *dst, int L, int W)
{
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < W; j++) {
            dst[(size_t)(i + 1) * (W + 2) + (j + 1)] = src[(size_t)i * W + j];
        }
    }
    for (int i = 0; i < L; i++) {
        dst[(size_t)(i + 1) * (W + 2) + 0] = src[(size_t)i * W];
        dst[(size_t)(i + 1) * (W + 2) + (W + 1)] = src[(size_t)i * W + W - 1];
    }
    for (int j = 0; j < W; j++) {
        dst[0 * (W + 2) + (j + 1)] = src[0 * W + j];
        dst[(size_t)(L + 1) * (W + 2) + (j + 1)] = src[(size_t)(L - 1) * W + j];
    }
    dst[0] = src[0];
    dst[W + 1] = src[W - 1];
    dst[(size_t)(L + 1) * (W + 2)] = src[(size_t)(L - 1) * W];
    dst[(size_t)(L + 1) * (W + 2) + (W + 1)] = src[(size_t)(L - 1) * W + W - 1];
}

int grid_init(grid *g, const char *fixture_dir)
{
    memset(g, 0, sizeof *g);
    char *cfg_path = join_path(fixture_dir, "config.json");
    jval *cfg = json_parse_file(cfg_path);
    if (!cfg) {
        fprintf(stderr, "pydeltarcm_c: cannot parse %s\n", cfg_path);
        free(cfg_path);
        return -1;
    }
    free(cfg_path);

    g->L = (int)json_get_int(cfg, "L", 0);
    g->W = (int)json_get_int(cfg, "W", 0);
    if (g->L <= 0 || g->W <= 0) {
        fprintf(stderr, "pydeltarcm_c: config missing L/W\n");
        json_free(cfg);
        return -1;
    }
    size_t n = (size_t)g->L * g->W;
    size_t padn = (size_t)(g->L + 2) * (g->W + 2);

    g->dx = json_get_num(cfg, "_dx", 0);
    g->dt = json_get_num(cfg, "_dt", 0);
    g->gamma = json_get_num(cfg, "_gamma", 0);
    g->theta_water = json_get_num(cfg, "_theta_water", 1.0);
    g->theta_sand = json_get_num(cfg, "theta_sand", 2.0);
    g->theta_mud = json_get_num(cfg, "theta_mud", 1.0);
    g->u_max = json_get_num(cfg, "u_max", 0);
    g->U_dep_mud = json_get_num(cfg, "U_dep_mud", 0);
    g->U_ero_mud = json_get_num(cfg, "U_ero_mud", 0);
    g->U_ero_sand = json_get_num(cfg, "U_ero_sand", 0);
    g->f_bedload = json_get_num(cfg, "_f_bedload", 0);
    g->C0_percent = json_get_num(cfg, "_C0_percent", 0);
    g->dry_depth = json_get_num(cfg, "dry_depth", 0);
    g->S0 = json_get_num(cfg, "_S0", 0);
    g->h0 = json_get_num(cfg, "_h0", 0);
    g->hb = json_get_num(cfg, "_hb", 0);
    g->H_SL = json_get_num(cfg, "_H_SL", 0);
    g->SLR = json_get_num(cfg, "_SLR", 0);
    g->omega_flow = json_get_num(cfg, "_omega_flow", 0);
    g->omega_sfc = json_get_num(cfg, "_omega_sfc", 0);
    g->omega_flow_iter = json_get_num(cfg, "omega_flow_iter", 0);
    g->alpha = json_get_num(cfg, "_alpha", 0);
    g->Csmooth = json_get_num(cfg, "_Csmooth", 0);
    g->lambda = json_get_num(cfg, "_lambda", 0);
    g->beta = json_get_num(cfg, "_beta", 0);
    g->stepmax = json_get_num(cfg, "_stepmax", 0);
    g->Vp_sed = json_get_num(cfg, "Vp_sed", 0);
    g->dVs = json_get_num(cfg, "dVs", 0);
    g->qw0 = json_get_num(cfg, "qw0", 0);
    g->qs0 = json_get_num(cfg, "qs0", 0);
    g->u0 = json_get_num(cfg, "_u0", 0);
    g->Qw0 = json_get_num(cfg, "Qw0", 0);
    g->Qs0 = json_get_num(cfg, "Qs0", 0);
    g->V0 = json_get_num(cfg, "V0", 0);
    g->Qp_water = json_get_num(cfg, "Qp_water", 0);
    g->g = json_get_num(cfg, "g", 9.81);
    g->diffusion_multiplier = json_get_num(cfg, "diffusion_multiplier", 0);
    g->CTR = json_get_num(cfg, "CTR", 0);
    g->active_layer_thickness = json_get_num(cfg, "_active_layer_thickness", 0);
    g->sand_frac_bc = json_get_num(cfg, "_sand_frac_bc", 0);
    g->subsidence_rate = json_get_num(cfg, "_subsidence_rate", 0);
    g->start_subsidence = json_get_num(cfg, "_start_subsidence", 0);
    g->time = 0.0;
    g->seed = (uint32_t)json_get_int(cfg, "_seed", 0);

    g->itermax = (int)json_get_int(cfg, "_itermax", 1);
    g->Np_water = (int)json_get_int(cfg, "_Np_water", 0);
    g->Np_sed = (int)json_get_int(cfg, "_Np_sed", 0);
    g->N0 = (int)json_get_int(cfg, "N0", 0);
    g->L0 = (int)json_get_int(cfg, "L0", 0);
    g->N_crossdiff = (int)json_get_int(cfg, "N_crossdiff", 0);
    g->Nsmooth = (int)json_get_int(cfg, "_Nsmooth", 0);
    g->toggle_subsidence = json_get_bool(cfg, "_toggle_subsidence", 0);
    g->force_deposit = json_get_bool(cfg, "_force_deposit", 0);
    g->time_iter = 0;

    char statedir_leaf[64];
    snprintf(statedir_leaf, sizeof statedir_leaf, "state");
    char *state_dir = join_path(fixture_dir, statedir_leaf);

    /* float32 state fields (all loaded from the state directory) */
    g->eta0 = load_state_f32(state_dir, "eta0.npy", n);
    g->eta = load_state_f32(state_dir, "eta.npy", n);
    g->depth = load_state_f32(state_dir, "depth.npy", n);
    g->stage = load_state_f32(state_dir, "stage.npy", n);
    /* Hnew is a per-update temporary in the reference (not serialized) */
    g->Hnew = calloc(n, sizeof(float));
    if (!g->Hnew) {
        fprintf(stderr, "pydeltarcm_c: out of memory (Hnew)\n");
        return -1;
    }
    g->qw = load_state_f32(state_dir, "qw.npy", n);
    g->qx = load_state_f32(state_dir, "qx.npy", n);
    g->qy = load_state_f32(state_dir, "qy.npy", n);
    g->qxn = load_state_f32(state_dir, "qxn.npy", n);
    g->qyn = load_state_f32(state_dir, "qyn.npy", n);
    g->qwn = load_state_f32(state_dir, "qwn.npy", n);
    g->uw = load_state_f32(state_dir, "uw.npy", n);
    g->ux = load_state_f32(state_dir, "ux.npy", n);
    g->uy = load_state_f32(state_dir, "uy.npy", n);
    g->qs = load_state_f32(state_dir, "qs.npy", n);
    g->Vp_dep_sand = load_state_f32(state_dir, "Vp_dep_sand.npy", n);
    g->Vp_dep_mud = load_state_f32(state_dir, "Vp_dep_mud.npy", n);
    g->sand_frac = load_state_f32(state_dir, "sand_frac.npy", n);
    g->active_layer = load_state_f32(state_dir, "active_layer.npy", n);
    g->sfc_sum = load_state_f32(state_dir, "sfc_sum.npy", n);
    g->sfc_visit = load_state_f32(state_dir, "sfc_visit.npy", n);
    g->mod_water_weight = load_state_f32(state_dir, "mod_water_weight.npy", n);
    g->mod_sed_weight = load_state_f32(state_dir, "mod_sed_weight.npy", n);
    g->mod_stable_weight = load_state_f32(state_dir, "mod_stable_weight.npy", n);
    g->mod_erosion = load_state_f32(state_dir, "mod_erosion.npy", n);

    /* integer state */
    g->cell_type = load_state_i64(state_dir, "cell_type.npy", n);
    char *inlet_path = join_path(state_dir, "inlet.npy");
    size_t nn = 0;
    g->inlet = npy_load_i64(inlet_path, &nn, NULL, NULL, 0);
    free(inlet_path);
    if (!g->inlet) {
        fprintf(stderr, "pydeltarcm_c: cannot load inlet.npy\n");
        exit(1);
    }
    g->n_inlet = (int)nn;

    /* padded buffers (contents refilled per routing pass) */
    g->pad_eta = calloc(padn, sizeof(float));
    g->pad_depth = calloc(padn, sizeof(float));
    g->pad_cell_type = calloc(padn, sizeof(int64_t));
    if (!g->pad_eta || !g->pad_depth || !g->pad_cell_type) {
        fprintf(stderr, "pydeltarcm_c: out of memory (pads)\n");
        return -1;
    }

    /* derived walk tables */
    load_table_f32(state_dir, "ivec_flat.npy", g->ivec);
    load_table_f32(state_dir, "jvec_flat.npy", g->jvec);
    load_table_f32(state_dir, "distances_flat.npy", g->distances);
    load_table_i64(state_dir, "iwalk_flat.npy", g->iwalk);
    load_table_i64(state_dir, "jwalk_flat.npy", g->jwalk);
    load_table_i64(state_dir, "ravel_walk_flat.npy", g->ravel_walk_flat);
    load_table_i64(state_dir, "kernel1.npy", g->kernel1);
    load_table_i64(state_dir, "kernel2.npy", g->kernel2);

    /* water weights */
    g->water_weights = calloc((size_t)n * 9, sizeof(double));
    if (!g->water_weights) {
        fprintf(stderr, "pydeltarcm_c: out of memory (water_weights)\n");
        return -1;
    }

    mt_seed(&g->rng, g->seed);

    free(state_dir);
    json_free(cfg);
    return 0;
}

void grid_free(grid *g)
{
    free(g->eta);
    free(g->eta0);
    free(g->depth);
    free(g->stage);
    free(g->Hnew);
    free(g->qw);
    free(g->qx);
    free(g->qy);
    free(g->qxn);
    free(g->qyn);
    free(g->qwn);
    free(g->uw);
    free(g->ux);
    free(g->uy);
    free(g->qs);
    free(g->Vp_dep_sand);
    free(g->Vp_dep_mud);
    free(g->sand_frac);
    free(g->active_layer);
    free(g->sfc_sum);
    free(g->sfc_visit);
    free(g->mod_water_weight);
    free(g->mod_sed_weight);
    free(g->mod_stable_weight);
    free(g->mod_erosion);
    free(g->cell_type);
    free(g->inlet);
    free(g->pad_eta);
    free(g->pad_depth);
    free(g->pad_cell_type);
    free(g->water_weights);
    memset(g, 0, sizeof *g);
}
