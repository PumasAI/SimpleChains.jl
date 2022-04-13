# Small Multi-Layer Perceptron

Here, we'll fit a simple network:
```julia
using SimpleChains

mlpd = SimpleChain(
  static(4),
  TurboDense(tanh, 32),
  TurboDense(tanh, 16),
  TurboDense(identity, 4)
)
```

Our goal here will be to try and approximate the matrix exponential:
```julia
function f(x)
  N = Base.isqrt(length(x))
  A = reshape(view(x, 1:N*N), (N,N))
  expA = exp(A)
  vec(expA)
end

T = Float32;
X = randn(T, 2*2, 1_000);
Y = reduce(hcat, map(f, eachcol(X)));
```

Now, to train our network:
```julia
@time p = SimpleChains.init_params(mlpd);

G = SimpleChains.alloc_threaded_grad(mlpd);

mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y));
mlpdloss(X, p)

@time SimpleChains.train_batched!(
  G, p, mlpdloss, X, SimpleChains.ADAM(1e-6), 1_000_000, batchsize = size(X)[end]
);
mlpdloss(X, p)
```

