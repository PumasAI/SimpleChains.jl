
# Elementwise transforms
"""
    Activation(activation)

Applies `activation` function elementwise.
"""
struct Activation{F}
  f::F
end
parameter_free(::Activation) = true
numparam(::Activation, id) = static(0), id
init_params!(::Activation, p, id, ::AbstractRNG) = p, id
_check_input_dims(::Activation, _) = nothing

forward_layer_output_size(::Val{T}, a::Activation, s) where {T} =
  align(prod(s) * static_sizeof(T)), s

Base.show(io::IO, a::Activation) = print(io, "Activation layer applying: ", a.f)

function (a::Activation)(x::AbstractArray{T}, p::Ptr, pu::Ptr{UInt8}) where {T}
  f = a.f
  C = PtrArray(Ptr{T}(pu), static_size(x))
  pu += length(C) * sizeof(T)
  @turbo for i ∈ eachindex(x)
    C[i] = f(x[i])
  end
  C, p, pu
end
function call!(x::AbstractArray, a::Activation, p::Ptr, pu::Ptr{UInt8})
  f = a.f
  @turbo for i ∈ eachindex(x)
    x[i] = f(x[i])
  end
  x, p, pu
end

function valgrad_layer!(
  pg::Ptr{T},
  a::Activation,
  x,
  p::Ptr{T},
  pu::Ptr{UInt8}
) where {T}
  ∂C = PtrArray(Ptr{T}(pu), static_size(x))
  pu += length(∂C) * sizeof(T)
  C = PtrArray(Ptr{T}(pu), static_size(x))
  pu += length(C) * sizeof(T)
  _valgrad_layer!(∂C, C, pg, a, x, p, pu)
end
function _valgrad_layer!(
  ∂C,
  C,
  pg::Ptr{T},
  a::Activation,
  x,
  p::Ptr{T},
  pu::Ptr{UInt8}
) where {T}
  ∂f = ∂(a.f)
  @turbo for i ∈ eachindex(x)
    C[i], ∂C[i] = ∂f(x[i])
  end
  pg, C, p, pu
end
@inline pullback_param!(__::Ptr, ::Activation, C̄, B, p::Ptr, pu::Ptr{UInt8}) =
  nothing
function pullback_arg!(
  ::Activation,
  C̄,
  _,
  ::Ptr{T},
  pu::Ptr{UInt8},
  pu2::Ptr{UInt8}
) where {T}
  ∂C = PtrArray(Ptr{T}(pu), static_size(C̄))
  @turbo for i ∈ eachindex(∂C)
    C̄[i] *= ∂C[i]
  end
  C̄, pu2
end

# specialization for identity
function (::Activation{typeof(identity)})(
  x::AbstractArray,
  p::Ptr,
  pu::Ptr{UInt8}
)
  return x, p, pu
end
call!(
  x::AbstractArray,
  ::Activation{typeof(identity)},
  p::Ptr,
  pu::Ptr{UInt8}
) = x, p, pu
function valgrad_layer!(
  pg::Ptr{T},
  ::Activation{typeof(identity)},
  x,
  p::Ptr{T},
  pu::Ptr{UInt8}
) where {T}
  pg, x, p, pu
end
function _valgrad_layer!(
  __,
  _,
  pg::Ptr{T},
  ::Activation{typeof(identity)},
  x,
  p::Ptr{T},
  pu::Ptr{UInt8}
) where {T}
  pg, x, p, pu
end
function pullback_arg!(
  ::Activation{typeof(identity)},
  C̄,
  _,
  ::Ptr{T},
  ::Ptr{UInt8},
  pu2::Ptr{UInt8}
) where {T}
  C̄, pu2
end

fast_fuse(::typeof(relu)) = True()
fast_fuse(::typeof(abs)) = True()
fast_fuse(::typeof(abs2)) = True()
fast_fuse(::typeof(Base.FastMath.abs_fast)) = True()
fast_fuse(::typeof(Base.FastMath.abs2_fast)) = True()
fast_fuse(::typeof(identity)) = True()
fast_fuse(_) = False()

const σ = SLEEFPirates.sigmoid_fast
