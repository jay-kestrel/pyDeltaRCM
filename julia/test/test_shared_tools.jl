using Test
using DeltaRCMJulia
const DJ = DeltaRCMJulia

@testset "npy reader round-trips the committed fixture" begin
    # Python wrote eta[0, 3] = -1.0 and eta[9, 9] = -1.0 (init_tools probes);
    # julia (W, L) layout reads them at M[j+1, i+1].
    st_meta = joinpath(FIXTURE_DIR, "state")
    eta = DJ.npyread(joinpath(st_meta, "eta.npy"))
    @test size(eta) == (10, 10) && eltype(eta) == Float32
    @test eta[4, 1] == -1.0f0   # python (i=0, j=3)
    @test eta[10, 10] == -1.0f0 # python (i=9, j=9)

    cell_type = DJ.npyread(joinpath(st_meta, "cell_type.npy"))
    @test eltype(cell_type) == Int64
    inlet = DJ.npyread(joinpath(st_meta, "inlet.npy"))
    @test size(inlet) == (3,)  # committed fixture: three inlet columns
end

@testset "custom_ravel/custom_unravel docstring vectors" begin
    @test DJ.custom_unravel(5124, 100, 200) == (25, 124)
    @test DJ.custom_ravel(25, 124, 100, 200) == 5124
    # exhaustive round-trip on the fixture geometry
    ok = all(DJ.custom_ravel(DJ.custom_unravel(f0, 10, 10)..., 10, 10) == f0
             for f0 in 0:99)
    @test ok
    @test DJ.custom_unravel(0, 10, 10) == (0, 0)
    @test DJ.custom_unravel(99, 10, 10) == (9, 9)
end

@testset "get_steps per direction code" begin
    st = load_water_state()   # walk arrays straight from the fixture
    for d0 in 0:8
        dist, istep, jstep, astep = DJ.get_steps([Int64(d0)], st.iwalk_flat,
                                                 st.jwalk_flat)
        # iwalk/jwalk per python: iwalk = [[-1,0,1],[-1,0,1],[-1,0,1]].ravel()
        # jwalk = [[1,1,1],[0,0,0],[-1,-1,-1]].ravel(); diagonal codes get
        # the sqrt(2) distance, cardinal/center codes 1.0/0.0
        iw = st.iwalk_flat[d0 + 1]
        jw = st.jwalk_flat[d0 + 1]
        want_dist = (iw == 0 || jw == 0) ? 1.0 : Float64(sqrt(2.0))
        if d0 == 4
            want_dist = 0.0
        end
        @test dist[1] == want_dist
        @test istep[1] == iw && jstep[1] == jw
        @test astep[1] == (iw != 0 || jw != 0)
    end
    # center code 4: no movement, zero distance, not active
    dist, istep, jstep, astep = DJ.get_steps([Int64(4)], st.iwalk_flat,
                                             st.jwalk_flat)
    @test dist[1] == 0.0 && !astep[1] && istep[1] == 0 && jstep[1] == 0
end

@testset "translation audit: sentinel and C-order layout" begin
    st = load_water_state()

    # sentinel: inactive parcel (flat 0) picks direction 4 and draws nothing
    mutable struct CountingRNG
        draws::Int
    end
    function DJ.get_random_uniform(c::CountingRNG, limit::Real)
        c.draws += 1
        return limit * 0.5
    end
    ww = zeros(Float64, 9, st.W, st.L)
    ww[:, :, :] .= 1.0 / 9.0  # uniform: u = 0.5 * 1.0 lands mid-bin 4, no ties
    crng = CountingRNG(0)
    dirs = DJ._choose_next_directions(crng, Int64[0, 42], ww, st.L, st.W)
    @test dirs == [4, 4]
    @test crng.draws == 1  # only the active parcel consumed a draw

    # C-order ravel: python flat index (i0, j0) = i0*W + j0 reads julia M[j0+1, i0+1]
    # from the dumped eta: eta[0, 3] == -1.0 -> M[4, 1]
    @test st.eta[3 + 1, 0 + 1] == DJ.npyread(joinpath(FIXTURE_DIR, "state", "eta.npy"))[4, 1]

    # walk matrix: python (Np, size_indices) -> julia (size_indices, Np)
    ref = load_reference(1)
    @test ndims(ref["free_surf_walk_inds"]) == 2
    @test size(ref["free_surf_walk_inds"], 2) == st.Np_water
end
