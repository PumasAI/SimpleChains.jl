# SimpleChains

<!-- hy-mt2-i18n:start -->
[English](./README.md) | [中文](./README_zh-CN.md) | [日本語](./README_ja.md) | **Español**
<!-- hy-mt2-i18n:end -->


[![Estable](https://img.shields.io/badge/docs-stable-blue.svg)](https://PumasAI.github.io/SimpleChains.jl/stable)
[![Desarrollo](https://img.shields.io/badge/docs-dev-blue.svg)](https://PumasAI.github.io/SimpleChains.jl/dev)
[![CI](https://github.com/PumasAI/SimpleChains.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/PumasAI/SimpleChains.jl/actions/workflows/CI.yml)
[![codecov-img](https://codecov.io/gh/PumasAI/SimpleChains.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PumasAI/SimpleChains.jl)

`SimpleChains.jl` solo soporta cadenas simples, pero está diseñado para ser rápido en el CPU para problemas pequeños. Actualmente, `valgrad!` es el único método para extraer información de gradiente.

```julia
using SimpleChains, BenchmarkTools

# 24 covariables por cada 200 observaciones
x = rand(24, 200); # 24 entradas por 200 observaciones

# 2 salidas por cada 200 observaciones
y = Matrix{Float64}(undef, 2, 200).= randn.().* 10;

schain = SimpleChain(
  static(24), # dimensión de la entrada (opcional)
  TurboDense{true}(tanh, 8), # capa densa con sesgo que mapea a 8 salidas y aplica la activación `tanh`
  SimpleChains.Dropout(0.2), # capa de dropout
  TurboDense{false}(identity, 2), # capa densa sin sesgo que mapea a 2 salidas y usa la activación `identity`
  SquaredLoss(y)
);

p = SimpleChains.init_params(schain)
g = similar(p);

# Evaluación completamente in situ
@benchmark valgrad!($g, $schain, $x, $p) # dropout activo
```
Para comparación, usando Flux, escribiríamos:
```julia
using Flux

chain = Chain(
  Dense(24, 8, tanh; bias = true),
  Flux.Dropout(0.2),
  Dense(8, 2, identity; bias = false)
);
chain.layers[2].active = true # activar el dropout

ya = Array(y);

@benchmark gradient(Flux.params($chain)) do
  Flux.mse($chain($x), $ya)
end
```

Resultados del benchmark:
```julia
julia> @benchmark valgrad!($g, $schain, $x, $p) # dropout activo
BechmarkTools.Trial: 10000 muestras con 6 evaluaciones.
 Rango (mínimo … máximo):  5.274 μs …  33.075 μs  ┊ GC (mínimo … máximo): 0.00% … 0.00%
 Tiempo (mediana):     5.657 μs               ┊ GC (mediana):    0.00%
 Tiempo (media ± σ):   5.646 μs ± 349.777 ns  ┊ GC (media ± σ):  0.00% ± 0.00%
 Estimación de memoria: 0 bytes, estimación de asignaciones: 0.
  
julia> @benchmark gradient(Flux.params($chain)) do
         Flux.mse($chain($x), $ya)
       end
BechmarkTools.Trial: 10000 muestras con 1 evaluación.
 Rango (mínimo … máximo):   83.674 μs …   4.865 ms  ┊ GC (mínimo … máximo): 0.00% … 93.21%
 Tiempo (mediana):      96.430 μs               ┊ GC (mediana):    0.00%
 Tiempo (media ± σ):   106.897 μs ± 197.689 μs  ┊ GC (media ± σ):  7.96% ±  4.22%
 Estimación de memoria: 182.55 KiB, estimación de asignaciones: 316.
```

