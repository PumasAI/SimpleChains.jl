using StaticArrays: @SVector, SVector, MVector
using ForwardDiff
using Random: MersenneTwister

# Gradients are checked against ForwardDiff rather than against `ChainRules.rrule`: the
# rule delegates to the same hand written pullbacks, so comparing the two agrees even
# when both are wrong.
#
# Chains whose result is a mutable heap array (matrix/batched inputs, or outputs too
# large for the `SArray` path) cannot be differentiated, so they are not covered here
# beyond asserting the abort -- where it surfaces, and as which exception, has varied
# across Enzyme patch releases and Julia versions.

const ENZ_CHAIN = SimpleChain(
  static(3),
  TurboDense{true}(SimpleChains.tanh_fast, static(6)),
  TurboDense{true}(identity, static(2))
)
const ENZ_P =
  Array(SimpleChains.init_params(ENZ_CHAIN; rng = MersenneTwister(1234)))
const ENZ_U = Float32[0.3, -1.2, 0.7]
const ENZ_U2 = Float32[-0.5, 0.8, 0.1]
const ENZ_LOSSY =
  SimpleChains.add_loss(ENZ_CHAIN, SquaredLoss(Float32[0.5, -0.5]))

# A batched input, and its loss terminated chain, whose result is still a scalar.
const ENZ_U_B = Float32[0.3 -0.5 0.2 0.9; -1.2 0.8 0.4 -0.1; 0.7 0.1 -0.6 0.3]
const ENZ_LOSSY_B = SimpleChains.add_loss(
  ENZ_CHAIN,
  SquaredLoss(Float32[0.5 -0.2 0.1 0.4; -0.5 0.3 -0.1 0.2])
)

# One output past the `ol < 64` cutoff for the `SArray` path, so the result is a heap array.
const ENZ_WIDE = SimpleChain(
  static(3),
  TurboDense{true}(SimpleChains.tanh_fast, static(8)),
  TurboDense{true}(identity, static(64))
)
const ENZ_WIDE_P =
  Array(SimpleChains.init_params(ENZ_WIDE; rng = MersenneTwister(5)))

enz_sumsq(chain, x, q) = sum(abs2, chain(x, q))
enz_two(chain, x, y, q) = sum(abs2, chain(x, q)) + sum(abs2, chain(y, q))
enz_call(chain, x, q) = chain(x, q)
enz_vjpdot(chain, x, q, w) = sum(chain(x, q) .* w)
enz_scaled(chain, x, q) = 3.0f0 * chain(x, q)
function enz_inactive_1(chain, x, q)
  chain(x, q)
  return sum(abs2, q)
end
enz_inactive_2(chain, x, q) = (chain(x, q) > 0 ? 1 : 2) * sum(abs2, q)

ref_p = ForwardDiff.gradient(q -> enz_sumsq(ENZ_CHAIN, ENZ_U, q), ENZ_P)
ref_u = ForwardDiff.gradient(x -> enz_sumsq(ENZ_CHAIN, x, ENZ_P), ENZ_U)

@testset "active input and parameters" begin
  dp, du = zero(ENZ_P), zero(ENZ_U)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Duplicated(ENZ_U, du),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test dp ≈ ref_p
  @test du ≈ ref_u
end

@testset "static array input" begin
  dp = zero(ENZ_P)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Const(SVector{3,Float32}(ENZ_U)),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test dp ≈ ref_p
end

@testset "active static array input" begin
  us = SVector{3,Float32}(ENZ_U)
  λ = @SVector Float32[1, 1]
  dp = zero(ENZ_P)
  result = Enzyme.autodiff(
    Enzyme.Reverse,
    enz_vjpdot,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Active(us),
    Enzyme.Duplicated(ENZ_P, dp),
    Enzyme.Const(λ)
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
    Enzyme.Reverse,
    enz_two,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Const(ENZ_U),
    Enzyme.Const(ENZ_U2),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test dp ≈
        ForwardDiff.gradient(q -> enz_two(ENZ_CHAIN, ENZ_U, ENZ_U2, q), ENZ_P)
end

@testset "runtime activity" begin
  dp, du = zero(ENZ_P), zero(ENZ_U)
  Enzyme.autodiff(
    Enzyme.set_runtime_activity(Enzyme.Reverse),
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Duplicated(ENZ_U, du),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test dp ≈ ref_p
  @test du ≈ ref_u
end

@testset "constant parameters" begin
  du = zero(ENZ_U)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Duplicated(ENZ_U, du),
    Enzyme.Const(ENZ_P)
  )
  @test du ≈ ref_u
end

# Batched immutable returns arrive as one `Active` annotation per batch rather than
# as a single `Active`.
@testset "batched parameters" begin
  da, db = zero(ENZ_P), zero(ENZ_P)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Const(ENZ_U),
    Enzyme.BatchDuplicated(ENZ_P, (da, db))
  )
  @test da ≈ ref_p
  @test db ≈ ref_p
end

@testset "accumulates into a nonzero shadow" begin
  dp = copy(ref_p)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Const(ENZ_U),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test dp ≈ 2 .* ref_p
end

# `ChainRules.rrule` routes loss terminated chains through `ElementwisePullback`,
# which yields no input tangent at all; the rule uses `valgrad!((ga, gp), ...)` so
# the input gradient is available.
@testset "loss terminated chain" begin
  dp, du = zero(ENZ_P), zero(ENZ_U)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_call,
    Enzyme.Active,
    Enzyme.Const(ENZ_LOSSY),
    Enzyme.Duplicated(ENZ_U, du),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test dp ≈ ForwardDiff.gradient(q -> ENZ_LOSSY(ENZ_U, q), ENZ_P)
  @test du ≈ ForwardDiff.gradient(x -> ENZ_LOSSY(x, ENZ_P), ENZ_U)
  @test ChainRules.rrule(ENZ_LOSSY, ENZ_U, ENZ_P)[2](one(Float32))[2] isa
        ChainRules.NoTangent
end

# An `Active` input together with a width greater than one is the only way to reach the
# batched `arg_tangent` branch, which returns one tangent per batch instead of one.
@testset "batched parameters with an active input" begin
  us = SVector{3,Float32}(ENZ_U)
  λ = @SVector Float32[1, 1]
  da, db = zero(ENZ_P), zero(ENZ_P)
  result = Enzyme.autodiff(
    Enzyme.Reverse,
    enz_vjpdot,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Active(us),
    Enzyme.BatchDuplicated(ENZ_P, (da, db)),
    Enzyme.Const(λ)
  )
  ref_dot_u =
    ForwardDiff.gradient(x -> enz_vjpdot(ENZ_CHAIN, x, ENZ_P, λ), ENZ_U)
  ref_dot_p = ForwardDiff.gradient(q -> enz_vjpdot(ENZ_CHAIN, us, q, λ), ENZ_P)
  @test result[1][2] isa Tuple{SVector{3,Float32},SVector{3,Float32}}
  @test all(t -> t ≈ ref_dot_u, result[1][2])
  @test da ≈ ref_dot_p
  @test db ≈ ref_dot_p
end

# The only path that reaches `valgrad!` with the argument Enzyme handed the rule: a
# scalar seed and an active input. `valgrad!` has no `pullback_arg!` for a
# `StaticArray`, so the rule has to materialize the argument first.
@testset "loss terminated chain with a static array input" begin
  us = SVector{3,Float32}(ENZ_U)
  dp = zero(ENZ_P)
  result = Enzyme.autodiff(
    Enzyme.Reverse,
    enz_call,
    Enzyme.Active,
    Enzyme.Const(ENZ_LOSSY),
    Enzyme.Active(us),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test result[1][2] ≈ ForwardDiff.gradient(x -> ENZ_LOSSY(x, ENZ_P), ENZ_U)
  @test dp ≈ ForwardDiff.gradient(q -> ENZ_LOSSY(us, q), ENZ_P)
end

# The no-loss path goes through `valgrad_noloss`, which never materializes the
# argument, so an `MVector` has to work there on its own terms -- both when it is
# differentiated and when it is not.
@testset "MVector input on a chain without a loss" begin
  um = MVector{3,Float32}(ENZ_U)
  @testset "constant input" begin
    dp = zero(ENZ_P)
    Enzyme.autodiff(
      Enzyme.Reverse,
      enz_sumsq,
      Enzyme.Active,
      Enzyme.Const(ENZ_CHAIN),
      Enzyme.Const(um),
      Enzyme.Duplicated(ENZ_P, dp)
    )
    @test dp ≈ ref_p
  end
  @testset "active input" begin
    du, dp = zero(um), zero(ENZ_P)
    Enzyme.autodiff(
      Enzyme.Reverse,
      enz_sumsq,
      Enzyme.Active,
      Enzyme.Const(ENZ_CHAIN),
      Enzyme.Duplicated(um, du),
      Enzyme.Duplicated(ENZ_P, dp)
    )
    @test du ≈ ref_u
    @test dp ≈ ref_p
  end
end

# An `MVector` is pointer backed, unlike an `SVector`, so it reaches `valgrad!` on the
# scalar path without being copied out first.
@testset "loss terminated chain with an MVector input" begin
  um = MVector{3,Float32}(ENZ_U)
  du, dp = zero(um), zero(ENZ_P)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_call,
    Enzyme.Active,
    Enzyme.Const(ENZ_LOSSY),
    Enzyme.Duplicated(um, du),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test du ≈ ForwardDiff.gradient(x -> ENZ_LOSSY(x, ENZ_P), ENZ_U)
  @test dp ≈ ForwardDiff.gradient(q -> ENZ_LOSSY(ENZ_U, q), ENZ_P)
end

# Every other loss terminated test uses the chain's result directly, so the seed is
# exactly 1 and the rule's `gp .*= seed` rescaling never runs. Scaling the result is
# what makes the seed something else.
@testset "loss terminated chain with a non-unit seed" begin
  du, dp = zero(ENZ_U), zero(ENZ_P)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_scaled,
    Enzyme.Active,
    Enzyme.Const(ENZ_LOSSY),
    Enzyme.Duplicated(ENZ_U, du),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test dp ≈ ForwardDiff.gradient(q -> enz_scaled(ENZ_LOSSY, ENZ_U, q), ENZ_P)
  @test du ≈ ForwardDiff.gradient(x -> enz_scaled(ENZ_LOSSY, x, ENZ_P), ENZ_U)
end

# An inactive input on a loss terminated chain is the only way to reach the
# parameter-only `valgrad!(gp, ...)` branch; every other inactive-input test returns a
# vector, so it takes the `valgrad_noloss` path where the flag is unused.
@testset "loss terminated chain with a constant input" begin
  dp = zero(ENZ_P)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_call,
    Enzyme.Active,
    Enzyme.Const(ENZ_LOSSY),
    Enzyme.Const(ENZ_U),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test dp ≈ ForwardDiff.gradient(q -> ENZ_LOSSY(ENZ_U, q), ENZ_P)
end

# It is the chain's result, not its input, that decides whether the rule is reachable:
# a batched input through a loss terminated chain still returns a scalar.
@testset "batched input on a loss terminated chain" begin
  dU, dp = zero(ENZ_U_B), zero(ENZ_P)
  Enzyme.autodiff(
    Enzyme.Reverse,
    enz_call,
    Enzyme.Active,
    Enzyme.Const(ENZ_LOSSY_B),
    Enzyme.Duplicated(ENZ_U_B, dU),
    Enzyme.Duplicated(ENZ_P, dp)
  )
  @test dp ≈ ForwardDiff.gradient(q -> ENZ_LOSSY_B(ENZ_U_B, q), ENZ_P)
  @test dU ≈ ForwardDiff.gradient(x -> ENZ_LOSSY_B(x, ENZ_P), ENZ_U_B)
end

# A result that reaches nothing differentiable gets a `Const` return annotation, whether it
# is discarded (so no primal is asked for either) or only compared against (primal still
# needed, derivative still not).
@testset "inactively used result" begin
  @testset "result completely unused" begin
    dp = zero(ENZ_P)
    Enzyme.autodiff(
      Enzyme.Reverse,
      enz_inactive_1,
      Enzyme.Active,
      Enzyme.Const(ENZ_CHAIN),
      Enzyme.Const(ENZ_U),
      Enzyme.Duplicated(ENZ_P, dp)
    )
    @test dp ≈
          ForwardDiff.gradient(q -> enz_inactive_1(ENZ_CHAIN, ENZ_U, q), ENZ_P)
  end
  @testset "result used in a non-differentiable way" begin
    dp = zero(ENZ_P)
    Enzyme.autodiff(
      Enzyme.Reverse,
      enz_inactive_2,
      Enzyme.Active,
      Enzyme.Const(ENZ_LOSSY),
      Enzyme.Const(ENZ_U),
      Enzyme.Duplicated(ENZ_P, dp)
    )
    @test dp ≈
          ForwardDiff.gradient(q -> enz_inactive_2(ENZ_LOSSY, ENZ_U, q), ENZ_P)
  end
  @testset "nonzero input shadow left unchanged" begin
    du, dp = fill(3.0f0, length(ENZ_U)), zero(ENZ_P)
    Enzyme.autodiff(
      Enzyme.Reverse,
      enz_inactive_1,
      Enzyme.Active,
      Enzyme.Const(ENZ_CHAIN),
      Enzyme.Duplicated(ENZ_U, du),
      Enzyme.Duplicated(ENZ_P, dp)
    )
    @test all(isequal(3), du)
    @test dp ≈
          ForwardDiff.gradient(q -> enz_inactive_1(ENZ_CHAIN, ENZ_U, q), ENZ_P)
  end
  @testset "active input set to zero" begin
    us = SVector{3,Float32}(ENZ_U)
    dp = zero(ENZ_P)
    result = Enzyme.autodiff(
      Enzyme.Reverse,
      enz_inactive_1,
      Enzyme.Active,
      Enzyme.Const(ENZ_CHAIN),
      Enzyme.Active(us),
      Enzyme.Duplicated(ENZ_P, dp)
    )
    @test result[1][2] isa SVector{3,Float32}
    @test all(iszero, result[1][2])
    @test dp ≈
          ForwardDiff.gradient(q -> enz_inactive_1(ENZ_CHAIN, ENZ_U, q), ENZ_P)
  end
  # Enzyme runs a discarded call for a function without a rule, so an invalid one has to
  # error here as well, even though no primal is asked for
  @testset "invalid call with an unused result" begin
    @test_throws "Input argument" Enzyme.autodiff(
      Enzyme.Reverse,
      enz_inactive_1,
      Enzyme.Active,
      Enzyme.Const(ENZ_CHAIN),
      Enzyme.Const(Float32[0.3, -1.2, 0.7, 0.5]),
      Enzyme.Duplicated(ENZ_P, zero(ENZ_P))
    )
  end
end

@testset "the chain itself must be Const" begin
  dchain = Enzyme.make_zero(ENZ_LOSSY)
  @test iszero(SimpleChains.target(last(dchain.layers)))
  @test_throws "chain itself to be `Enzyme.Const`" Enzyme.autodiff(
    Enzyme.Reverse,
    enz_call,
    Enzyme.Active,
    Enzyme.Duplicated(ENZ_LOSSY, dchain),
    Enzyme.Const(ENZ_U),
    Enzyme.Duplicated(ENZ_P, zero(ENZ_P))
  )
end

# `SVector` parameters are not supported
# see https://github.com/PumasAI/SimpleChains.jl/issues/224
@testset "SVector parameters are rejected" begin
  ps = SVector{length(ENZ_P),Float32}(ENZ_P)
  msg = @static VERSION ≥ v"1.11" ? "pointer-backed" : "conversion to pointer"
  @test_throws msg Enzyme.autodiff(
    Enzyme.Reverse,
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Const(ENZ_U),
    Enzyme.Active(ps)
  )
  @test_throws msg Enzyme.autodiff(
    Enzyme.Reverse,
    enz_call,
    Enzyme.Active,
    Enzyme.Const(ENZ_LOSSY),
    Enzyme.Duplicated(ENZ_U, zero(ENZ_U)),
    Enzyme.Active(ps)
  )
  # The parameter values are needed for the input gradient too,
  # so freezing them does not help
  @test_throws msg Enzyme.autodiff(
    Enzyme.Reverse,
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Duplicated(ENZ_U, zero(ENZ_U)),
    Enzyme.Const(ps)
  )
  # Even if the chain does not contribute to the gradient,
  # it may still need to be evaluated,
  # so that we need to check for `SVector` params
  @test_throws msg Enzyme.autodiff(
    Enzyme.Reverse,
    enz_inactive_2,
    Enzyme.Active,
    Enzyme.Const(ENZ_LOSSY),
    Enzyme.Const(ENZ_U),
    Enzyme.Active(ps)
  )
  forward, _reverse = Enzyme.autodiff_thunk(
    Enzyme.ReverseSplitWithPrimal,
    Enzyme.Const{typeof(enz_call)},
    Enzyme.Active,
    Enzyme.Const{typeof(ENZ_LOSSY)},
    Enzyme.Const{typeof(ENZ_U)},
    Enzyme.Active{typeof(ps)}
  )
  @test_throws msg forward(
    Enzyme.Const(enz_call),
    Enzyme.Const(ENZ_LOSSY),
    Enzyme.Const(ENZ_U),
    Enzyme.Active(ps)
  )
  # The chain runs even when nothing reads its result, so this errors too
  @test_throws msg Enzyme.autodiff(
    Enzyme.Reverse,
    enz_inactive_1,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Const(ENZ_U),
    Enzyme.Active(ps)
  )
end

# What matters is that an unsupported heap array result aborts loudly instead of
# returning a silently wrong gradient; if this ever starts passing, the limitation
# notes are stale. The concrete exception is Enzyme internal churn -- a `BoundsError`
# in its rule setup on Enzyme 0.13.196/Julia 1.12.6, a `MethodError` from its generic
# `sum` wrapper on 0.13.199 -- so only the throw itself is asserted.
@testset "heap array results still abort" begin
  @test ENZ_CHAIN(ENZ_U_B, ENZ_P) isa SimpleChains.StrideArraysCore.StrideArray
  @test_throws Exception Enzyme.autodiff(
    Enzyme.Reverse,
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_CHAIN),
    Enzyme.Duplicated(ENZ_U_B, zero(ENZ_U_B)),
    Enzyme.Duplicated(ENZ_P, zero(ENZ_P))
  )
  # A static output of 64 is one past the `ol < 64` cutoff for the `SArray` path.
  @test ENZ_WIDE(ENZ_U, ENZ_WIDE_P) isa
        SimpleChains.StrideArraysCore.StrideArray
  @test_throws Exception Enzyme.autodiff(
    Enzyme.Reverse,
    enz_sumsq,
    Enzyme.Active,
    Enzyme.Const(ENZ_WIDE),
    Enzyme.Duplicated(ENZ_U, zero(ENZ_U)),
    Enzyme.Duplicated(ENZ_WIDE_P, zero(ENZ_WIDE_P))
  )
end
