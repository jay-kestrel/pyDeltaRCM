using Test
using DeltaRCMJulia
const DJ = DeltaRCMJulia

"""
Run the full water phase for one update against the Python reference dump.

The fixture is the post-init state; update 1 ran at `time_iter = 0` (so the
`omega_sfc` blend is skipped and `qx/qy` bind to the first-iteration result)
and update 2 ran at `time_iter = 1` on update 1's outputs.
"""
function run_reference_update(update::Int)
    st = load_water_state()
    ref = load_reference(update)
    st.time_iter = update - 1
    if update > 1
        prior = load_reference(update - 1)
        # hydrate the fields update `update` reads, per the dumper's contract
        for name in ("qxn", "qyn", "qwn", "qx", "qy", "qw", "stage", "depth",
                     "uw", "ux", "uy")
            setproperty!(st, Symbol(name), prior[name])
        end
    end
    rng = DJ.MT19937()
    DJ.set_rng_state!(rng, ref["rng_before"]...)
    DJ.route_water!(st, rng)
    return st, ref, rng
end

@testset "route_water end-to-end: update 1 (time_iter = 0)" begin
    st, ref, rng = run_reference_update(1)
    for name in ("qxn", "qyn", "qwn", "qx", "qy", "qw", "stage", "depth",
                 "uw", "ux", "uy", "free_surf_flag", "free_surf_walk_inds",
                 "sfc_visit", "sfc_sum", "Hnew")
        bit_identical(getproperty(st, Symbol(name)), ref[name], name)
    end
    pos, key = DJ.pack_rng_state(rng)
    @test pos == ref["rng_after"][1]
    @test key == ref["rng_after"][2]
end

@testset "route_water end-to-end: update 2 (time_iter = 1, omega_sfc blend)" begin
    st, ref, rng = run_reference_update(2)
    for name in ("qxn", "qyn", "qwn", "qx", "qy", "qw", "stage", "depth",
                 "uw", "ux", "uy", "free_surf_flag", "free_surf_walk_inds",
                 "sfc_visit", "sfc_sum", "Hnew")
        bit_identical(getproperty(st, Symbol(name)), ref[name], name)
    end
    pos, key = DJ.pack_rng_state(rng)
    @test pos == ref["rng_after"][1]
    @test key == ref["rng_after"][2]
end
