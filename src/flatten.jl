

struct Flatten{N} end
Flatten(N) = Flatten{convert(Int,N)::Int}()
@generated _dec(::Flatten{N}) where {N} = Flatten{N-1}()

init_params!(::Flatten, p, id) = p, id
parameter_free(::Flatten) = true

getoutputdim(::Flatten{1}, inputdim) = inputdim
function getoutputdim(::Flatten{N}, inputdim) where {N}
  d0 = first(inputdim)
  t0 = Base.tail(inputdim)
  d1 = first(t0)
  t1 = Base.tail(t0)
  getoutputdim(_dec(Flatten{N}()), (d0*d1, t1...))
end

function output_size(::Val{T}, ::Flatten{N}, inputdim::Tuple) where {T,N}
  0, getoutputdim(Flatten{N}(), inputdim)
end

numparam(::Flatten{N}, inputdim) where {N} = 0, getoutputdim(Flatten{N}(), inputdim)

function (::Flatten{N})(A::AbstractArray{T}, p::Ptr, pu::Ptr{UInt8}) where {T,N}
  reshape(A, getoutputdim(Flatten{N}(), size(A))), p, pu
end
function valgrad_layer!(pg::Ptr, ::Flatten{N}, A, p, pu) where {N}
  B, p, pu = Flatten{N}()(A, p, pu)
  return pg, B, p, pu
end
function pullback!(::Ptr, ::Flatten, B̄, A, p, pu, pu2)
  return reshape(B̄, size(A)), pu2
end


