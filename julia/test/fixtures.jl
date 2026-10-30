# Test fixtures: hydrate a WaterState from the committed bench fixture and
# read the Python-reference dumps for comparison.

using Test
using DeltaRCMJulia
const DJ = DeltaRCMJulia

const REPO_ROOT = dirname(dirname(@__DIR__))
const FIXTURE_DIR = joinpath(REPO_ROOT, "bench", "fixture_default_small")
const REF_DIR = joinpath(@__DIR__, "fixtures", "reference")

"""
    read_flat_config(path) -> Dict{String, Float64}

Minimal reader for the fixture's flat scalar `config.json` (numbers/strings
only; nested structures fail loudly). Keeps the test dependency set empty.
"""
function read_flat_config(path::AbstractString)
    cfg = Dict{String,Float64}()
    for m in eachmatch(r"\"([^\"]+)\"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"s,
                       read(path, String))
        cfg[m.captures[1]] = parse(Float64, m.captures[2])
    end
    length(cfg) > 0 || error("no flat scalar entries parsed from $path")
    return cfg
end

const OUTPUT_FIELDS = (
    "qxn", "qyn", "qwn", "qx", "qy", "qw", "stage", "depth", "uw", "ux", "uy",
    "free_surf_flag", "free_surf_walk_inds", "sfc_visit", "sfc_sum", "Hnew",
)

"""Hydrate a WaterState from the committed bench fixture (post-init state)."""
function load_water_state(fixture_dir::AbstractString=FIXTURE_DIR)
    cfg = read_flat_config(joinpath(fixture_dir, "config.json"))
    sd = joinpath(fixture_dir, "state")
    rd(name) = DJ.npyread(joinpath(sd, "$name.npy"))
    return DJ.WaterState(
        L=Int(cfg["L"]), W=Int(cfg["W"]), L0=Int(cfg["L0"]), CTR=Int(cfg["CTR"]),
        itermax=Int(cfg["_itermax"]), Np_water=Int(cfg["_Np_water"]),
        dx=cfg["_dx"], Qp_water=cfg["Qp_water"], stepmax=Int(cfg["_stepmax"]),
        H_SL=cfg["_H_SL"], omega_sfc=cfg["_omega_sfc"],
        omega_flow=cfg["_omega_flow"], omega_flow_iter=cfg["omega_flow_iter"],
        Nsmooth=Int(cfg["_Nsmooth"]), Csmooth=cfg["_Csmooth"],
        u0=cfg["_u0"], h0=cfg["_h0"], S0=cfg["_S0"], gamma=cfg["_gamma"],
        theta_water=cfg["_theta_water"], u_max=cfg["u_max"],
        dry_depth=cfg["dry_depth"], qw0=cfg["qw0"],
        eta=rd("eta"), stage=rd("stage"), depth=rd("depth"),
        qw=rd("qw"), qx=rd("qx"), qy=rd("qy"), qwn=rd("qwn"),
        qxn=rd("qxn"), qyn=rd("qyn"), uw=rd("uw"), ux=rd("ux"), uy=rd("uy"),
        cell_type=rd("cell_type"), mod_water_weight=rd("mod_water_weight"),
        inlet=rd("inlet"),
        iwalk_flat=rd("iwalk_flat"), jwalk_flat=rd("jwalk_flat"),
        ravel_walk_flat=rd("ravel_walk_flat"), ivec_flat=rd("ivec_flat"),
        jvec_flat=rd("jvec_flat"), distances_flat=rd("distances_flat"),
        free_surf_walk_inds=zeros(Int64, Int(cfg["size_indices"]),
                                  Int(cfg["_Np_water"])),
    )
end

"""Load one update's Python-reference dump (fields + per-iteration arrays)."""
function load_reference(update::Int, ref_dir::AbstractString=REF_DIR)
    ud = joinpath(ref_dir, "update$update")
    ref = Dict{String,Any}()
    for name in OUTPUT_FIELDS
        ref[name] = DJ.npyread(joinpath(ud, "$name.npy"))
    end
    ref["water_weights_iter0"] = DJ.npyread(joinpath(ud, "water_weights_iter0.npy"))
    ref["start_indices_iter0"] = DJ.npyread(joinpath(ud, "start_indices_iter0.npy"))
    ref["Hsmth"] = DJ.npyread(joinpath(ud, "Hsmth.npy"))
    ref["rng_before"] = (DJ.npyread(joinpath(ud, "rng_before_index.npy")),
                         DJ.npyread(joinpath(ud, "rng_before_key.npy")))
    ref["rng_after"] = (DJ.npyread(joinpath(ud, "rng_after_index.npy")),
                        DJ.npyread(joinpath(ud, "rng_after_key.npy")))
    return ref
end

"""Bit-exact comparison with a debuggable failure message."""
function bit_identical(a::AbstractArray, b::AbstractArray, label::AbstractString)
    size(a) == size(b) ||
        return (@test false; "shape mismatch for $label: $(size(a)) vs $(size(b))")
    Ta, Tb = eltype(a), eltype(b)
    for idx in eachindex(a)
        ok = (Ta <: AbstractFloat || Tb <: AbstractFloat) ? isequal(a[idx], b[idx]) :
             a[idx] == b[idx]
        ok || begin
            println("first mismatch at $label[$(Tuple(idx))]: got " *
                    "$(a[idx]) ($(typeof(a[idx]))), want $(b[idx]) ($(typeof(b[idx])))")
            return (@test false; false)
        end
    end
    @test true
    return true
end
