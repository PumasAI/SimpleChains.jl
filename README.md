# SimpleChains

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSIMD.github.io/SimpleChains.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSIMD.github.io/SimpleChains.jl/dev)
[![Build Status](https://github.com/JuliaSIMD/SimpleChains.jl/workflows/CI/badge.svg)](https://github.com/JuliaSIMD/SimpleChains.jl/actions)


`SimpleChains.jl` only supports simple chains, but it intends to be fast for small problems on the CPU.

```julia
using SimpleChains, BenchmarkTools

# 24 covariates each per 200 observations
x = rand(24, 200); # 24 inputs per 200 observations

# 2 responses each per 200 observations
y = StrideArray{Float64}(undef, (static(2),200)) .= randn.() .* 10;

schain = SimpleChain((
  TurboDense{true}(tanh, (static(24),static(8))), # 24 x 8 dense layer with bias and `tanh` activation
  SimpleChains.Dropout(0.2), # dropout layer
  TurboDense{false}(identity, (static(8),static(2))), # 8 x 2 dense layer without bias and `identity` activation
  SquaredLoss(y) # squared error loss function
));

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
julia> # Entirely in place evaluation
       @benchmark valgrad!($g, $schain, $x, $p) # dropout active
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     7.254 μs (0.00% GC)
  median time:      7.318 μs (0.00% GC)
  mean time:        7.328 μs (0.00% GC)
  maximum time:     18.084 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     4
  
julia> @benchmark gradient(Flux.params($chain)) do
         Flux.mse($chain($x), $ya)
       end
BenchmarkTools.Trial:
  memory estimate:  184.52 KiB
  allocs estimate:  368
  --------------
  minimum time:     121.267 μs (0.00% GC)
  median time:      126.493 μs (0.00% GC)
  mean time:        138.602 μs (6.95% GC)
  maximum time:     5.276 ms (95.47% GC)
  --------------
  samples:          10000
  evals/sample:     1
```

