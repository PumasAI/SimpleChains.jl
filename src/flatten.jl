
"""
    Flatten{N}()

Flattens the first `N` dimensions. E.g.,

```julia
julia> Flatten{2}()(rand(2,3,4))
6×4 Matrix{Float64}:
 0.0609115  0.597285  0.279899  0.888223
 0.0667422  0.315741  0.351003  0.805629
 0.678297   0.350817  0.984215  0.399418
 0.125801   0.566696  0.96873   0.57744
 0.331961   0.350742  0.59598   0.741998
 0.26345    0.144635  0.076433  0.330475
```
"""
struct Flatten{N} end
Flatten(N) = Flatten{convert(Int, N)::Int}()
@generated _dec(::Flatten{N}) where {N} = Flatten{N - 1}()

parameter_free(::Flatten) = true

getoutputdim(::Flatten{1}, inputdim) = inputdim
function getoutputdim(::Flatten{N}, inputdim) where {N}
  d0 = first(inputdim)
  t0 = Base.tail(inputdim)
  d1 = first(t0)
  t1 = Base.tail(t0)
  getoutputdim(_dec(Flatten{N}()), (d0 * d1, t1...))
end

function forward_layer_output_size(::Val{T}, ::Flatten{N}, inputdim::Tuple) where {T,N}
  static(0), getoutputdim(Flatten{N}(), inputdim)
end

init_params!(::Flatten{N}, p, id, ::AbstractRNG) where {N} = p, getoutputdim(Flatten{N}(), id)


numparam(::Flatten{N}, inputdim) where {N} = 0, getoutputdim(Flatten{N}(), inputdim)

@inline function (::Flatten{N})(A::AbstractArray) where {N}
  reshape(A, getoutputdim(Flatten{N}(), static_size(A)))
end
@inline function (::Flatten{N})(A::AbstractArray, p::Ptr, pu::Ptr{UInt8}) where {N}
  Flatten{N}()(A), p, pu
end
function valgrad_layer!(pg::Ptr, ::Flatten{N}, A, p, pu) where {N}
  B, p, pu = Flatten{N}()(A, p, pu)
  return pg, B, p, pu
end
function pullback!(::Ptr, ::Flatten, B̄, A, p::Ptr, pu::Ptr{UInt8}, pu2::Ptr{UInt8})
  return reshape(B̄, static_size(A)), pu2
end
