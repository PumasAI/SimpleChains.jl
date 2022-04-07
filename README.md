# SimpleChains

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSIMD.github.io/SimpleChains.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSIMD.github.io/SimpleChains.jl/dev)
[![Build Status](https://github.com/JuliaSIMD/SimpleChains.jl/workflows/CI/badge.svg)](https://github.com/JuliaSIMD/SimpleChains.jl/actions)
[codecov-img]:          https://codecov.io/gh/PumasAI/SimpleChains.jl/branch/master/graph/badge.svg           "Code Coverage"

`SimpleChains.jl` only supports simple chains, but it intends to be fast for small problems on the CPU.
Currently, `valgrad!` is the only means of extracting gradient information.

```julia
using SimpleChains, BenchmarkTools

# 24 covariates each per 200 observations
x = rand(24, 200); # 24 inputs per 200 observations

# 2 responses each per 200 observations
y = StrideArray{Float64}(undef, (static(2),200)) .= randn.() .* 10;

schain = SimpleChain(
  static(24), # input dimension (optional)
  (
    TurboDense{true}(tanh, static(8)), # dense layer with bias that maps to 8 outputs and applies `tanh` activation
    SimpleChains.Dropout(0.2), # dropout layer
    TurboDense{false}(identity, static(2)), # dense layer without bias that maps to 2 outputs and `identity` activation
    SquaredLoss(y)
  ) # squared error loss function
);

p = randn(SimpleChains.numparam(schain)); # something like glorot would probably be a better way to initialize
g = similar(p);

# Entirely in place evaluation
@benchmark valgrad!($g, $schain, $x, $p) # dropout active
```
For comparison, using Flux, we would write:
```julia
using Flux

chain = Chain(
  Dense(24, 8, tanh; bias = true),
  Flux.Dropout(0.2),
  Dense(8, 2, identity; bias = false)
);
chain.layers[2].active = true # activate dropout

ya = Array(y);

@benchmark gradient(Flux.params($chain)) do
  Flux.mse($chain($x), $ya)
end
```

Benchmark results:
```julia
julia> @benchmark valgrad!($g, $schain, $x, $p) # dropout active
BechmarkTools.Trial: 10000 samples with 6 evaluations.
 Range (min … max):  5.274 μs …  33.075 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.657 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.646 μs ± 349.777 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%
 Memory estimate: 0 bytes, allocs estimate: 0.
  
julia> @benchmark gradient(Flux.params($chain)) do
         Flux.mse($chain($x), $ya)
       end
BechmarkTools.Trial: 10000 samples with 1 evaluations.
 Range (min … max):   83.674 μs …   4.865 ms  ┊ GC (min … max): 0.00% … 93.21%
 Time  (median):      96.430 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   106.897 μs ± 197.689 μs  ┊ GC (mean ± σ):  7.96% ±  4.22%
 Memory estimate: 182.55 KiB, allocs estimate: 316.
```

