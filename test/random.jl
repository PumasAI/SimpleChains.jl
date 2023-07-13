
using Random, VectorizedRNG

_mean(x) = sum(x) / length(x)
function _mean_std(x, xbar = _mean(x))
  xbar, sqrt(sum(abs2 ∘ Base.Fix2(-, xbar), x) / (length(x) - 1))
end
x = Vector{Float64}(undef, 2047);
y = Vector{Float64}(undef, 2047);

vrng = VectorizedRNG.MutableXoshift(3);
rng = VERSION >= v"1.7" ? Random.Xoshiro(3) : Random.MersenneTwister(4);

SimpleChains.glorot_uniform!(x, vrng);
SimpleChains.glorot_uniform!(y, rng);
mx, sx = _mean_std(x)
@test abs(mx) < 0.01
@test sx ≈ 0.03125 rtol = 1e-2
my, sy = _mean_std(y)
@test abs(my) < 0.01
@test sy ≈ 0.03125 rtol = 1e-2

SimpleChains.glorot_normal!(x, vrng);
SimpleChains.glorot_normal!(y, rng);
mx, sx = _mean_std(x)
@test abs(mx) < 0.01
@test sx ≈ 0.03125 rtol = 1e-2
my, sy = _mean_std(y)
@test abs(my) < 0.01
@test sy ≈ 0.03125 rtol = 1e-2
