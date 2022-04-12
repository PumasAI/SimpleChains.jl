using VectorizedRNG

"""
  Dropout(p) # 0 < p < 1

Dropout layer.

When evaluated without gradients, it multiplies inputs by `(1 - p)`.
When evaluated with gradients, it randomly zeros `p` proportion of inputs.
"""
struct Dropout{R<:Union{Nothing,VectorizedRNG.AbstractRNG}}
  p::UInt32
  rng::R
end
Dropout(x::T, rng = local_rng()) where {T<:Union{Float32,Float64}} =
  Dropout(Base.fptoui(UInt32, T(0xffffffff) * x), rng)

getrng(::Dropout{Nothing}) = local_rng()
getrng(d::Dropout{<:VectorizedRNG.AbstractRNG}) = getfield(d, :rng)

function Base.show(io::IO, d::Dropout)
  print(io, "Dropout(p=$(Float64(d.p)/0xffffffff))")
end


gradval(::Val{T}, d::Dropout) where {T} = T(0xffffffff) / (T(0xffffffff) - d.p)
numparam(::Dropout, id) = 0, id
parameter_free(::Dropout) = true

init_params!(::Dropout, p, id) = p, id

function (d::Dropout)(B::AbstractVecOrMat{T}, p::Ptr, pu::Ptr{UInt8}) where {T}
  x = muladd(T(d.p), -inv(T(typemax(UInt32))), one(T))
  @turbo for i ∈ eachindex(B)
    B[i] *= x
  end
  B, p, pu # inference
end


getpcmp(::StaticInt{W}, ::StaticInt{W}, x) where {W} = x
getpcmp(::StaticInt{W}, ::StaticInt{WU}, x) where {W,WU} =
  getpcmp(StaticInt(W), StaticInt(WU), x, Static.gt(StaticInt(W), StaticInt(WU)))
function getpcmp(::StaticInt{W}, ::StaticInt{WU}, x::UInt32, ::True) where {W,WU}
  (x >> 16) % UInt16
end
function getpcmp(::StaticInt{W}, ::StaticInt{WU}, x::UInt32, ::False) where {W,WU}
  (x % UInt64) << 32
end

layer_output_size(::Val{T}, d::Dropout, s) where {T} = align((prod(s) + 7) & -8), s

function valgrad_layer!(pg::Ptr{T}, d::Dropout, x, p::Ptr{T}, pu::Ptr{UInt8}) where {T}
  si = StrideIndex{1,(1,),1}((StaticInt(1),), (StaticInt(0),))

  N = static_length(x)
  m = PtrArray(stridedpointer(reinterpret(Ptr{Bit}, pu), si), (N,), Val((true,)))
  rng = getrng(d)
  _pcmp = d.p
  state = VectorizedRNG.getstate(rng, Val{2}(), pick_vector_width(UInt64))

  GC.@preserve x begin
    ptrx = VectorizedRNG.zero_pointer(x)
    ptrm = VectorizedRNG.zero_pointer(m)
    W = (pick_vector_width(T) * pick_vector_width(UInt64)) ÷ pick_vector_width(Float64)
    W2 = W + W
    WU = pick_vector_width(UInt32)
    pcmp = getpcmp(W, WU, _pcmp)
    # n = MM(W, 0)
    n = 0
    while n < VectorizedRNG.vadd(N, 1 - 2W)
      state, zvu2 = VectorizedRNG.random_unsigned(state, Val{2}(), UInt64)
      m₂ = reinterpret(typeof(pcmp), zvu2) > pcmp
      u₂ = Unroll{1,Int(W),2,1,Int(W),0x0000000000000000,1}((n,))
      vstore!(ptrx, vload(ptrx, u₂, m₂), u₂)
      vstore!(ptrm, m₂, u₂)
      n = VectorizedRNG.vadd(W2, n)
    end
    if n < VectorizedRNG.vsub(N, W)
      msk = VectorizationBase.mask(W, N)
      state, zvu2 = VectorizedRNG.random_unsigned(state, Val{2}(), UInt64)
      m₂ = reinterpret(typeof(pcmp), zvu2) > pcmp
      m₂msk = m₂ & msk
      u₂ = Unroll{1,Int(W),2,1,Int(W),0x0000000000000000,1}((n,))
      vstore!(ptrx, vload(ptrx, u₂, m₂msk), u₂, msk)
      vstore!(ptrm, m₂, u₂)
    elseif n < N
      msk = VectorizationBase.mask(W, N)
      state, _zvu1 = VectorizedRNG.random_unsigned(state, Val{1}(), UInt64)
      (z₁,) = VectorizedRNG.data(_zvu1)
      m₁ = reinterpret(typeof(pcmp), z₁) > pcmp
      m₁msk = m₁ & msk
      u₁ = (MM(W, n),)
      vstore!(ptrx, vload(ptrx, u₁, m₁msk), u₁, msk)
      vstore!(ptrm, m₁, u₁)
    end
    VectorizedRNG.storestate!(rng, state)
  end # GC preserve  

  pg, x, p, align(pu + ((7 + N) & -8))
end
@inline pullback_param!(::Ptr, ::Dropout, _, __, ::Ptr, ::Ptr{UInt8}) = nothing

function pullback!(
  ::Ptr{T},
  ::Dropout,
  C̄,
  B,
  ::Ptr{T},
  pu::Ptr{UInt8},
  pu2::Ptr{UInt8},
) where {T}
  N = static_length(C̄)
  si = StrideIndex{1,(1,),1}((StaticInt(1),), (StaticInt(1),))
  m = PtrArray(stridedpointer(reinterpret(Ptr{Bit}, pu), si), (N,), Val((true,)))
  @turbo for n ∈ eachindex(m)
    C̄[n] = m[n] ? C̄[n] : zero(C̄[n])
  end
  C̄, pu2# returns `pu2` because we don't know where `C̄` was allocated
end

isstochastic(::Dropout) = true
