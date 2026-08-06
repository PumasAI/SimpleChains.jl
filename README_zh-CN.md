# SimpleChains

<!-- hy-mt2-i18n:start -->
[English](./README.md) | **中文** | [日本語](./README_ja.md) | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->


[![稳定版](https://img.shields.io/badge/docs-stable-blue.svg)](https://PumasAI.github.io/SimpleChains.jl/stable)
[![开发版](https://img.shields.io/badge/docs-dev-blue.svg)](https://PumasAI.github.io/SimpleChains.jl/dev)
[![持续集成](https://github.com/PumasAI/SimpleChains.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/PumasAI/SimpleChains.jl/actions/workflows/CI.yml)
[![Codecov覆盖率](https://codecov.io/gh/PumasAI/SimpleChains.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PumasAI/SimpleChains.jl)

`SimpleChains.jl` 仅支持简单链结构，但其设计初衷是在 CPU 上快速处理小型问题。目前，`valgrad!` 是获取梯度信息的唯一方法。

```julia
using SimpleChains, BenchmarkTools

# 每200个观测值对应24个协变量
x = rand(24, 200); # 每200个观测值有24个输入

# 每200个观测值对应2个响应值
y = Matrix{Float64}(undef, 2, 200).= randn.().* 10;

schain = SimpleChain(
  static(24), # 输入维度（可选）
  TurboDense{true}(tanh, 8), # 带偏置的密集层，输出为8个值并应用`tanh`激活函数
  SimpleChains.Dropout(0.2), # Dropout层
  TurboDense{false}(identity, 2), # 不带偏置的密集层，输出为2个值并应用`identity`激活函数
  SquaredLoss(y)
);

p = SimpleChains.init_params(schain)
g = similar(p);

# 完全在原位置进行评估
@benchmark valgrad!($g, $schain, $x, $p) # 启用Dropout
```
作为对比，使用Flux时代码如下：
```julia
using Flux

chain = Chain(
  Dense(24, 8, tanh; bias = true),
  Flux.Dropout(0.2),
  Dense(8, 2, identity; bias = false)
);
chain.layers[2].active = true # 启用Dropout

ya = Array(y);

@benchmark gradient(Flux.params($chain)) do
  Flux.mse($chain($x), $ya)
end
```

基准测试结果：
```julia
julia> @benchmark valgrad!($g, $schain, $x, $p) # dropout active
BechmarkTools.Trial: 10000个样本，进行6次评估。
范围（最小值……最大值）：5.274 μs……33.075 μs  ┊ GC（最小值……最大值）：0.00%……0.00%
时间（中位数）：5.657 μs               ┊ GC（中位数）：0.00%
时间（均值±标准差）：5.646 μs ± 349.777 ns  ┊ GC（均值±标准差）：0.00% ± 0.00%
内存估算值：0字节，分配量估算值：0。
  
julia> @benchmark gradient(Flux.params($chain)) do
         Flux.mse($chain($x), $ya)
       end
BechmarkTools.Trial：10000个样本，进行1次评估。
范围（最小值……最大值）：83.674 μs……4.865 ms  ┊ GC（最小值……最大值）：0.00%……93.21%
时间（中位数）：96.430 μs               ┊ GC（中位数）：0.00%
时间（均值±标准差）：106.897 μs ± 197.689 μs  ┊ GC（均值±标准差）：7.96% ± 4.22%
内存估算值：182.55 KiB，分配量估算值：316。
```

