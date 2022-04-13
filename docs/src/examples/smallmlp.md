# Small Multi-Layer Perceptron

Here, we'll fit a simple network with dropout layers:
```julia
using SimpleChains

mlpd = SimpleChain(
	static(10),
	TurboDense(tanh, 32),
	TurboDense(tanh, 16),
	TurboDense(identity, 2)
)
```

This isn't necessarilly a good architecture, nor one that's necessarilly suited for the example problem:
```julia
function f(x)
    T = eltype(x)
    s = log(x'x+T(0.1))
    2log(s*s+T(0.1)) -3, 1/2s
end

T = Float32;
X = randn(T, 10, 100_000);
Y = reinterpret(reshape, T, map(f, eachcol(X)));# .+ 0.1 .* randn.();
```

Now, to train our network:
```julia
@time p = SimpleChains.init_params(mlpd);
G = similar(p, length(p), min(Threads.nthreads(), (Sys.CPU_THREADS ÷ ((Sys.ARCH === :x86_64) + 1))));

mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y));
mlpdloss(X, p)

@time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(1e-6), 1_000);
mlpdloss(X, p)

```

