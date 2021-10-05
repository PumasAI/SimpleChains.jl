

# Elementwise transforms

struct Activation{F}
  f::F
end
parameter_free(::Activation) = true
numparam(::Activation) = 0
init_params!(::Activation, p) = p

output_size(::Val{T}, a::Activation, s) where {T} = align(prod(s)*(2sizeof(T))), s

function (a::Activation)(x::AbstractArray{T}, p::Ptr, pu::Ptr{UInt8}) where {T}
  f = a.f
  C = PtrArray(reinterpret(Ptr{T}, pu), size(x))
  pu += length(C)*sizeof(T)
  @turbo for i ∈ eachindex(x)
    C[i] = f(x[i])
  end
  C, p, pu
end

function valgrad_layer!(pg::Ptr{T}, a::Activation, x, p::Ptr{T}, pu::Ptr{UInt8}) where {T}
  ∂f = ∂(a.f)
  ∂C = PtrArray(reinterpret(Ptr{T}, pu), size(x))
  pu += length(∂C)*sizeof(T)
  C = PtrArray(reinterpret(Ptr{T}, pu), size(x))
  pu += length(C)*sizeof(T)
  @turbo for i ∈ eachindex(x)
    C[i], ∂C[i] = ∂f(x[i])
  end
  pg, C, p, pu
end
@inline pullback_param!(pg::Ptr, ::Activation, C̄, B, p::Ptr, pu::Ptr{UInt8}) = nothing
function pullback!(pg::Ptr{T}, a::Activation, C̄, B, p::Ptr{T}, pu::Ptr{UInt8}, pu2::Ptr{UInt8}) where {T}
  ∂C = PtrArray(reinterpret(Ptr{T}, pu), size(C̄))
  @turbo for i ∈ eachindex(∂C)
    C̄[i] *= ∂C[i]
  end
  C̄, pu2
end

