# MNIST - Convolutions

First, we load the data using [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl):
```julia
using MLDatasets
xtrain3, ytrain0 = MLDatasets.MNIST.traindata(Float32);
xtest3, ytest0 = MLDatasets.MNIST.testdata(Float32);
size(xtest3)
# (28, 28, 60000)
extrema(ytrain0) # digits, 0,...,9
# (0, 9)
```
The covariate data (`x`) were named `3` as these are three-dimensional arrays, containing the height x width x number of images.
The training data are vectors indicating the digit.
```julia
xtrain4 = reshape(xtrain3, 28, 28, 1, :);
xtest4 = reshape(xtest3, 28, 28, 1, :);
ytrain1 = UInt32.(ytrain0 .+ 1);
ytest1 = UInt32.(ytest0 .+ 1);
```
SimpleChains' convolutional layers expect that we have a channels-in dimension, so we shape the images to be four dimensional
It also currently defaults to 1-based indexing for its categories, so we shift all categories by 1.

We now define our model, LeNet5:
```julia
using SimpleChains

lenet = SimpleChain(
  (static(28), static(28), static(1)),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),
  SimpleChains.MaxPool(2, 2),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),
  SimpleChains.MaxPool(2, 2),
  Flatten(3),
  TurboDense(SimpleChains.relu, 120),
  TurboDense(SimpleChains.relu, 84),
  TurboDense(identity, 10),
)

lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain1));
```
We define the inputs as being statically sized `(28,28,1)` images.
Specifying the input sizes allows these to be checked.
Making them static, which we can do either in our simple chain, or by adding
static sizing to the images themselves using a package like [StrideArrays.jl](https://github.com/JuliaSIMD/StrideArrays.jl)
or [HybridArrays.jl](git@github.com:JuliaArrays/HybridArrays.jl.git). These packages are recommended
for allowing you to mix dynamic and static sizes; the batch size should probably
be left dynamic, as you're unlikely to want to specialize code generation on this,
given that it is likely to vary, increasing compile times while being unlikely to
improve runtimes.

In `SimpleChains`, the parameters are not a part of the model, but live as a
separate vector that you can pass around to optimizers of your choosing.
If you specified the input size, you create a random initial parameter vector
corresponding to the model:
```julia
@time p = SimpleChains.init_params(lenet);
```
The convolutional layers are initialized with a Glorot (Xavier) uniform distribution,
while the dense layers are initialized with a Glorot (Xaviar) normal distribution.
Biases are initialized to zero.
Because the number of parameters can be a function of the input size, these must
be provided if you didn't specify input dimension. For example:
```julia
@time p = SimpleChains.init_params(lenet, size(xtrain4));
```

To allow training to use multiple threads, you can create a gradient matrix, with
a number of rows equal to the length of the parameter vector `p`, and one column
per thread. For example:
```julia
estimated_num_cores = (Sys.CPU_THREADS ÷ ((Sys.ARCH === :x86_64) + 1));
G = SimpleChains.alloc_threaded_grad(lenetloss);
```
Here, we're estimating that the number of physical cores is half the number of threads
on an `x86_64` system, which is true for most -- but not all!!! -- of them.
Otherwise, we're assuming it is equal to the number of threads. This is of course also
likely to be wrong, e.g. recent Power CPUs may have 4 or even 8 threads per core.
You may wish to change this, or use [Hwloc.jl](https://github.com/JuliaParallel/Hwloc.jl) for an accurate number.

Now that this is all said and done, we can train for `10` epochs using the `ADAM` optimizer
with a learning rate of `3e-4`, and then assess the accuracy and loss of both the training
and test data:
```julia
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)
```
Training for an extra 10 epochs should be fast on most systems. Performance is currently known
to be poor on the M1 (PRs welcome, otherwise we'll look into this eventually), but should be 
good/great on systems with AVX2/AVX512:
```julia
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)
```

