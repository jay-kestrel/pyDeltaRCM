using Test

# Includes are at top level on purpose: including inside a @testset block
# would make their definitions local to the testset scope.
include("fixtures.jl")

@testset "DeltaRCMJulia" begin
    include("test_mt19937.jl")
    include("test_shared_tools.jl")
    include("test_water_weights.jl")
    include("test_route_water.jl")
end
