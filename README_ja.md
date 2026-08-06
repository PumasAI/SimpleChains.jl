# SimpleChains

<!-- hy-mt2-i18n:start -->
[English](./README.md) | [中文](./README_zh-CN.md) | **日本語** | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->


[![安定版](https://img.shields.io/badge/docs-stable-blue.svg)](https://PumasAI.github.io/SimpleChains.jl/stable)
[![開発版](https://img.shields.io/badge/docs-dev-blue.svg)](https://PumasAI.github.io/SimpleChains.jl/dev)
[![CI](https://github.com/PumasAI/SimpleChains.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/PumasAI/SimpleChains.jl/actions/workflows/CI.yml)
[![codecovのバッジ](https://codecov.io/gh/PumasAI/SimpleChains.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PumasAI/SimpleChains.jl)

`SimpleChains.jl`はシンプルなチェーンのみをサポートしていますが、CPU上での小規模な問題に対して高速に動作することを目指しています。  
現時点では、勾配情報を取得する手段は`valgrad!`のみです。

```julia
using SimpleChains, BenchmarkTools

# 200件の観測値ごとに24個の共変量
x = rand(24, 200); # 200件の観測値ごとに24個の入力

# 200件の観測値ごとに2個の出力
y = Matrix{Float64}(undef, 2, 200).= randn.().* 10;

schain = SimpleChain(
  static(24), # 入力次元（省略可能）
  TurboDense{true}(tanh, 8), # バイアスを持つ密集層で、8個の出力にマッピングし`tanh`活性化関数を適用
  SimpleChains.Dropout(0.2), # Dropout層
  TurboDense{false}(identity, 2), # バイアスのない密集層で、2個の出力にマッピングし`identity`活性化関数を適用
  SquaredLoss(y)
);

p = SimpleChains.init_params(schain)
g = similar(p);

# 完全にインプレースでの評価
@benchmark valgrad!($g, $schain, $x, $p) # Dropout有効
```
比較のため、Fluxを使用する場合は次のように記述します：
```julia
using Flux

chain = Chain(
  Dense(24, 8, tanh; bias = true),
  Flux.Dropout(0.2),
  Dense(8, 2, identity; bias = false)
);
chain.layers[2].active = true # Dropoutを有効化

ya = Array(y);

@benchmark gradient(Flux.params($chain)) do
  Flux.mse($chain($x), $ya)
end
```

ベンチマーク結果：
```julia
julia> @benchmark valgrad!($g, $schain, $x, $p) # dropout active
BechmarkTools.Trial: 10000サンプルで6回の評価を実施。
範囲 (最小 … 最大): 5.274 μs … 33.075 μs  ┊ GC (最小 … 最大): 0.00% … 0.00%
時間 (中央値): 5.657 μs               ┊ GC (中央値): 0.00%
時間 (平均 ± σ): 5.646 μs ± 349.777 ns  ┊ GC (平均 ± σ): 0.00% ± 0.00%
メモリ使用量の見積み: 0バイト、割り当て量の見積み: 0。
  
julia> @benchmark gradient(Flux.params($chain)) do
         Flux.mse($chain($x), $ya)
       end
BechmarkTools.Trial: 10000サンプルで1回の評価を実施。
範囲 (最小 … 最大): 83.674 μs … 4.865 ms  ┊ GC (最小 … 最大): 0.00% … 93.21%
時間 (中央値): 96.430 μs               ┊ GC (中央値): 0.00%
時間 (平均 ± σ): 106.897 μs ± 197.689 μs  ┊ GC (平均 ± σ): 7.96% ± 4.22%
メモリ使用量の見積み: 182.55 KiB、割り当て量の見積み: 316。
```

