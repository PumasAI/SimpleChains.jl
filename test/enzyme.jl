using StaticArrays: @SVector, SVector
using ForwardDiff
using Random: MersenneTwister

# Gradients are checked against ForwardDiff rather than against `ChainRules.rrule`: the
# rule delegates to the same hand written pullbacks, so comparing the two agrees even
# when both are wrong.
#
# Chains whose result is a mutable heap array (matrix/batched inputs, or outputs too
# large for the `SArray` path) abort inside Enzyme's own custom rule setup with a
# `BoundsError` before this rule runs, so they are not covered here.
#
# Everything differentiated below is a top level function. A helper defined inside a
# `@testset` is a closure, which Enzyme cannot prove readonly, so it never reaches the
# custom rule.

const ENZ_CHAIN = SimpleChain(
    static(3),
    TurboDense{true}(SimpleChains.tanh_fast, static(6)),
    TurboDense{true}(identity, static(2))
)
const ENZ_P = Array(SimpleChains.init_params(ENZ_CHAIN; rng = MersenneTwister(1234)))
const ENZ_U = Float32[0.3, -1.2, 0.7]
const ENZ_U2 = Float32[-0.5, 0.8, 0.1]
const ENZ_LOSSY = SimpleChains.add_loss(ENZ_CHAIN, SquaredLoss(Float32[0.5, -0.5]))

enz_sumsq(chain, x, q) = sum(abs2, chain(x, q))
enz_two(chain, x, y, q) = sum(abs2, chain(x, q)) + sum(abs2, chain(y, q))
enz_call(chain, x, q) = chain(x, q)
enz_vjpdot(chain, x, q, w) = sum(chain(x, q) .* w)

@testset "Enzyme" begin
    ref_p = ForwardDiff.gradient(q -> enz_sumsq(ENZ_CHAIN, ENZ_U, q), ENZ_P)
    ref_u = ForwardDiff.gradient(x -> enz_sumsq(ENZ_CHAIN, x, ENZ_P), ENZ_U)

    @testset "active input and parameters" begin
        dp, du = zero(ENZ_P), zero(ENZ_U)
        Enzyme.autodiff(
            Enzyme.Reverse, enz_sumsq, Enzyme.Active, Enzyme.Const(ENZ_CHAIN),
            Enzyme.Duplicated(ENZ_U, du), Enzyme.Duplicated(ENZ_P, dp)
        )
        @test dp ≈ ref_p
        @test du ≈ ref_u
    end

    @testset "static array input" begin
        dp = zero(ENZ_P)
        Enzyme.autodiff(
            Enzyme.Reverse, enz_sumsq, Enzyme.Active, Enzyme.Const(ENZ_CHAIN),
            Enzyme.Const(SVector{3, Float32}(ENZ_U)), Enzyme.Duplicated(ENZ_P, dp)
        )
        @test dp ≈ ref_p
    end

    @testset "active static array input" begin
        us = SVector{3, Float32}(ENZ_U)
        λ = @SVector Float32[1, 1]
        dp = zero(ENZ_P)
        result = Enzyme.autodiff(
            Enzyme.Reverse, enz_vjpdot, Enzyme.Active, Enzyme.Const(ENZ_CHAIN),
            Enzyme.Active(us), Enzyme.Duplicated(ENZ_P, dp), Enzyme.Const(λ)
        )
        @test collect(result[1][2]) ≈
            ForwardDiff.gradient(x -> enz_vjpdot(ENZ_CHAIN, x, ENZ_P, λ), ENZ_U)
        @test dp ≈ ForwardDiff.gradient(q -> enz_vjpdot(ENZ_CHAIN, us, q, λ), ENZ_P)
    end

    # Enzyme runs every augmented pass before the first reverse pass. A pullback held
    # across that boundary reads `get_heap_memory`'s task local buffer, which the second
    # call has already overwritten.
    @testset "repeated calls to the same chain" begin
        dp = zero(ENZ_P)
        Enzyme.autodiff(
            Enzyme.Reverse, enz_two, Enzyme.Active, Enzyme.Const(ENZ_CHAIN),
            Enzyme.Const(ENZ_U), Enzyme.Const(ENZ_U2), Enzyme.Duplicated(ENZ_P, dp)
        )
        @test dp ≈ ForwardDiff.gradient(q -> enz_two(ENZ_CHAIN, ENZ_U, ENZ_U2, q), ENZ_P)
    end

    @testset "runtime activity" begin
        dp, du = zero(ENZ_P), zero(ENZ_U)
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse), enz_sumsq, Enzyme.Active,
            Enzyme.Const(ENZ_CHAIN), Enzyme.Duplicated(ENZ_U, du),
            Enzyme.Duplicated(ENZ_P, dp)
        )
        @test dp ≈ ref_p
        @test du ≈ ref_u
    end

    @testset "constant parameters" begin
        du = zero(ENZ_U)
        Enzyme.autodiff(
            Enzyme.Reverse, enz_sumsq, Enzyme.Active, Enzyme.Const(ENZ_CHAIN),
            Enzyme.Duplicated(ENZ_U, du), Enzyme.Const(ENZ_P)
        )
        @test du ≈ ref_u
    end

    # Batched immutable returns arrive as one `Active` annotation per batch rather than
    # as a single `Active`.
    @testset "batched parameters" begin
        da, db = zero(ENZ_P), zero(ENZ_P)
        Enzyme.autodiff(
            Enzyme.Reverse, enz_sumsq, Enzyme.Active, Enzyme.Const(ENZ_CHAIN),
            Enzyme.Const(ENZ_U), Enzyme.BatchDuplicated(ENZ_P, (da, db))
        )
        @test da ≈ ref_p
        @test db ≈ ref_p
    end

    @testset "accumulates into a nonzero shadow" begin
        dp = copy(ref_p)
        Enzyme.autodiff(
            Enzyme.Reverse, enz_sumsq, Enzyme.Active, Enzyme.Const(ENZ_CHAIN),
            Enzyme.Const(ENZ_U), Enzyme.Duplicated(ENZ_P, dp)
        )
        @test dp ≈ 2 .* ref_p
    end

    # `ChainRules.rrule` routes loss terminated chains through `ElementwisePullback`,
    # which yields no input tangent at all; the rule uses `valgrad!((ga, gp), ...)` so
    # the input gradient is available.
    @testset "loss terminated chain" begin
        dp, du = zero(ENZ_P), zero(ENZ_U)
        Enzyme.autodiff(
            Enzyme.Reverse, enz_call, Enzyme.Active, Enzyme.Const(ENZ_LOSSY),
            Enzyme.Duplicated(ENZ_U, du), Enzyme.Duplicated(ENZ_P, dp)
        )
        @test dp ≈ ForwardDiff.gradient(q -> ENZ_LOSSY(ENZ_U, q), ENZ_P)
        @test du ≈ ForwardDiff.gradient(x -> ENZ_LOSSY(x, ENZ_P), ENZ_U)
        @test ChainRules.rrule(ENZ_LOSSY, ENZ_U, ENZ_P)[2](one(Float32))[2] isa
            ChainRules.NoTangent
    end
end
