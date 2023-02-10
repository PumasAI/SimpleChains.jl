# Adding a custom loss layer

Loss functions like the `LogitCrossEntropyLoss` are defined for users to be able to quickly prototype models on new problems. However, sometimes there is a need to write one's own customized loss function. This example will walk through this process.

To show which functions need to be implemented for your own custom loss, this example will walk through implementing a `BinaryLogitCrossEntropyLoss`, which acts on a model with only a single output, and binary targets.

## Mathematical background for a Binary Cross Entropy Loss
Consider the following model:

```math
p_\theta(X_i) = \sigma (f(X_i)),
```
where ``X_i`` is the input features, ``\sigma`` is the sigmoid function, given by ``\sigma(x)=(1+e^{-x})^{-1}`` and ``f_\theta`` is some function mapping defined by your model, which is parameterized by parameters ``\theta``. The output of ``f_\theta (X_i)`` is called the "logit". The loss function we want to calculate is the following:

```math
L(\theta| X, Y) = -\sum_i \left [ Y_i\ln{p_\theta (X_i)} + (1-Y_i)\ln{(1-p_\theta (X_i))} \right ],
```
where ``Y_i`` is the true binary label of the ``i^\text{th}`` sample. In order to implement this custom loss, we have to know what the gradient of this loss function is, w.r.t the parameters:

```math
    \frac{{\partial }}{{\partial } \theta} L(\theta | X, Y) = -\sum_i  Y_i\frac{{\partial }}{{\partial } \theta}\ln{p_\theta (X_i)} + (1-Y_i)\frac{{\partial }}{{\partial } \theta}\ln (1-p_\theta (X_i)).
```

To simplify this calculation, we can use the fact that ``1-p_\theta (x)=p_\theta (-x)``, and ``\frac{{\partial }}{{\partial } \theta}\ln(p_\theta(X_i))=(1+e^{f_\theta(X_i)})^{-1} \frac{{\partial }}{{\partial } \theta} f_\theta(X_i)``. We are left with:

```math
\frac{{\partial }}{{\partial } \theta} L(\theta| X, Y) = -\sum_i \left [ (2Y_i - 1){\left (1+e^{(2Y_i-1)f_\theta(X_i)} \right )}^{-1} \right ] \frac{{\partial }}{{\partial } \theta} f_\theta(X_i).
```

We have managed to write the derivative of the loss function, in terms of the derivative of the model, independently for each sample. The important part of this equation is the multiplicand of the partial derivative term; this term is the partial gradient used for back-propagation. From this point, we can begin writing the code.

## Implementing a custom loss type

We start by importing `SimpleChains.jl` into the current namespace:
```julia
using SimpleChains
```

We can now define our own type, which is a subtype of `SimpleChains.AbstractLoss`:

```julia
struct BinaryLogitCrossEntropyLoss{T,Y<:AbstractVector{T}} <: SimpleChains.AbstractLoss{T}
    targets::Y
end
```

The function used to get the inner targets is called `target` and can be defined easily:
```julia
SimpleChains.target(loss::BinaryLogitCrossEntropyLoss) = loss.targets
(loss::BinaryLogitCrossEntropyLoss)(x::AbstractArray) = BinaryLogitCrossEntropyLoss(x)
```

Next, we define how to calculate the loss, given some logits:

```julia
function calculate_loss(loss::BinaryLogitCrossEntropyLoss, logits)
    y = loss.targets
    total_loss = zero(eltype(logits))
    for i in eachindex(y)
        p_i = inv(1 + exp(-logits[i]))
        y_i = y[i]
        total_loss -= y_i * log(p_i) + (1 - y_i) * (1 - log(p_i))
    end
    total_loss
end
function (loss::BinaryLogitCrossEntropyLoss)(previous_layer_output::AbstractArray{T}, p::Ptr, pu) where {T}
    total_loss = calculate_loss(loss, previous_layer_output)
    total_loss, p, pu
end
```

As the other loss functions do this, we should define some functions to say that we don't want any preallocated temporary arrays:
```julia
function SimpleChains.layer_output_size(::Val{T}, sl::BinaryLogitCrossEntropyLoss, s::Tuple) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end
function SimpleChains.forward_layer_output_size(::Val{T}, sl::BinaryLogitCrossEntropyLoss, s) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end
```

Finally, we define how to back-propagate the gradient from this loss function:

```julia
function SimpleChains.chain_valgrad!(
    __,
    previous_layer_output::AbstractArray{T},
    layers::Tuple{BinaryLogitCrossEntropyLoss},
    _::Ptr,
    pu::Ptr{UInt8},
) where {T}
    loss = getfield(layers, 1)
    total_loss = calculate_loss(loss, previous_layer_output)
    y = loss.targets

    # Store the backpropagated gradient in the previous_layer_output array.
    for i in eachindex(y)
        sign_arg = 2 * y[i] - 1
        # Get the value of the last logit
        logit_i = previous_layer_output[i]
        previous_layer_output[i] = -(sign_arg * inv(1 + exp(sign_arg * logit_i)))
    end

    return total_loss, previous_layer_output, pu
end
```

That's all! The way we can now use this loss function, just like any other:

```julia
using SimpleChains

model = SimpleChain(
    static(2),
    TurboDense(tanh, 32),
    TurboDense(tanh, 16),
    TurboDense(identity, 1)
)

batch_size = 64
X = rand(Float32, 2, batch_size)
Y = rand(Bool, batch_size)

parameters = SimpleChains.init_params(model);
gradients = SimpleChains.alloc_threaded_grad(model);

# Add the loss like any other loss type
model_loss = SimpleChains.add_loss(model, BinaryLogitCrossEntropyLoss(Y));

SimpleChains.valgrad!(gradients, model_loss, X, parameters)
```

Or alternatively, if you want to just train the parameters in full:
```julia
epochs = 100
SimpleChains.train_unbatched!(gradients, parameters, model_loss, X, SimpleChains.ADAM(), epochs); 
```
