using Test
using DeltaRCMJulia
const DJ = DeltaRCMJulia

@testset "MT19937 numba parity" begin
    rng = DJ.MT19937()
    DJ.mt_seed!(rng, 0)
    # pinned vectors from the Python probe (NumPy legacy stream)
    @test DJ.get_random_uniform(rng, 1.0) == 0.5488135039273248
    @test DJ.get_random_uniform(rng, 0.1) == 0.07151893663724194

    # state export/import: restore must continue the stream exactly
    pos, key = DJ.pack_rng_state(rng)
    @test pos == 4  # 0-based: four uniforms consumed
    @test key[1:4] == UInt32[2443250962, 1093594115, 1878467924, 2709361018]
    rng2 = DJ.MT19937()
    DJ.set_rng_state!(rng2, pos, key)
    @test DJ.get_random_uniform(rng2, 1.0) ==
          (rng === rng2 ? nothing : DJ.get_random_uniform(rng, 1.0))
    # seed 1 first draw (regression guard for the seeding table)
    rng3 = DJ.MT19937()
    DJ.mt_seed!(rng3, 1)
    @test DJ.get_random_uniform(rng3, 1.0) == 0.417022004702574
end

@testset "random_pick semantics" begin
    # deterministic draw injector: get_random_uniform(limit) = limit * u
    mutable struct MockUniform
        u::Float64
        last_limit::Float64
        draws::Int
    end
    MockUniform(u) = MockUniform(u, NaN, 0)
    function DJ.get_random_uniform(m::MockUniform, limit::Real)
        m.draws += 1
        m.last_limit = limit
        return limit * m.u
    end

    prob = Float64[0, 0, 0.25, 0, 0.5, 0, 0.25, 0, 0]  # cum: 0,0,.25,.25,.75,.75,1,1,1
    m = MockUniform(0.1)
    @test DJ.random_pick(m, prob) == 2          # searchsortedleft: first cum >= 0.1
    @test m.last_limit == 1.0 && m.draws == 1   # limit is the total weight
    m2 = MockUniform(0.0)
    @test DJ.random_pick(m2, prob) == 0         # u == 0 hits first zero cum
    m3 = MockUniform(0.74)
    @test DJ.random_pick(m3, prob) == 4
    # tied boundary at 0.75: side='left' lands on the FIRST 0.75 bin (code 4)
    @test DJ.random_pick(MockUniform(0.75), prob) == 4
    @test DJ.random_pick(MockUniform(0.76), prob) == 6
    @test DJ.random_pick(MockUniform(1.0), prob) == 6

    z = zeros(Float64, 9)
    @test_throws DomainError DJ.random_pick(MockUniform(0.5), z)
end
