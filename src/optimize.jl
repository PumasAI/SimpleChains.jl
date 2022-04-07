
abstract type AbstractOptimizer end

struct ADAM <: AbstractOptimizer
  η::Float64
  β::Tuple{Float64,Float64}
end

ADAM(η = 0.001) = ADAM(η, (0.9, 0.999))

function update!(o::ADAM, (mt, vt, βp), x, Δ)
  @unpack η, β = o

  β₁ = β[1]
  β₂ = β[2]
  βp₁ = βp[1]
  βp₂ = βp[2]
  # @inbounds @fastmath for i ∈ eachindex(Δ)
  @turbo for i ∈ eachindex(Δ)
    mt[i] = β₁ * mt[i] + (1 - β₁) * Δ[i]
    vt[i] = β₂ * vt[i] + (1 - β₂) * Δ[i]^2
    # Δ[i] =  mt[i] / (1 - βp₁) / (sqrt(vt[i] / (1 - βp₂)) + 1e-8) * η
    Δᵢ = η * mt[i] / ((1 - βp₁) * (sqrt(vt[i] / (1 - βp₂)) + 1e-8))
    Δ[i] = Δᵢ
    x[i] -= Δᵢ#Δ[i]
  end
  βp[1] = βp₁ * β₁
  βp[2] = βp₂ * β₂
  return
end
@inline optmemsize(opt::ADAM, p::AbstractVector{T}) where {T} =
  2align(sizeof(T) * length(p)) + align(1)
@inline function optmemory(opt::ADAM, p::AbstractVector{T}, pu::Ptr{UInt8}) where {T}
  memoff = align(sizeof(T) * length(p))
  mt = PtrArray(reinterpret(Ptr{T}, pu), (ArrayInterface.static_length(p),))
  pu += memoff
  vt = PtrArray(reinterpret(Ptr{T}, pu), (ArrayInterface.static_length(p),))
  @turbo for i ∈ eachindex(mt)
    mt[i] = 0
    vt[i] = 0
  end
  βp_doesnot_fit_at_end = sizeof(T) * length(p) + 16 > memoff
  pu_p_memoff = pu + memoff
  pβp = ifelse(βp_doesnot_fit_at_end, pu_p_memoff, pu + sizeof(T) * length(p))
  pu = pu_p_memoff
  βp = PtrArray(reinterpret(Ptr{Float64}, pβp), (static(2),))
  @unpack β = opt
  βp[1] = β[1]
  βp[2] = β[2]
  pu = ifelse(βp_doesnot_fit_at_end, pu + align(1), pu)
  return (mt, vt, βp), pu
end

function train_unbatched!(g, p, _chn::Chain, X, opt::AbstractOptimizer, t::AbstractArray)
  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  pen = getpenalty(_chn)
  @unpack layers, memory = chn
  fl = Base.front(layers)
  ll = last(layers)
  optoff = optmemsize(opt, p)
  sx = ArrayInterface.size(pX)
  resize_memory!(layers, memory, pX, optoff)
  optbuffer, pm = optmemory(opt, p, pointer(memory))
  GC.@preserve p g memory X begin
    pg = pointer(g)
    pp = pointer(p)
    for y ∈ t
      layers_y = (fl..., ll(y))
      chain_valgrad_entry!(pg, pX, layers_y, pp, pm)
      apply_penalty!(g, pen, p, sx)
      update!(opt, optbuffer, p, g)
    end
  end
  p
end
function train_unbatched!(g, p, _chn::Chain, X, opt::AbstractOptimizer, iters::Int)
  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  pen = getpenalty(_chn)
  sx = ArrayInterface.size(pX)
  @unpack layers, memory = chn
  optoff = optmemsize(opt, p)
  resize_memory!(layers, memory, pX, optoff)
  optbuffer, pm = optmemory(opt, p, pointer(memory))
  GC.@preserve p g memory X begin
    pg = pointer(g)
    pp = pointer(p)
    for _ ∈ 1:iters
      chain_valgrad_entry!(pg, pX, layers, pp, pm)
      apply_penalty!(g, pen, p, sx)
      update!(opt, optbuffer, p, g)
    end
  end
  p
end
function train_unbatched!(g, p, _chn::Chain, X, opt::AbstractOptimizer)
  t = target(_chn)
  if _iterate_over_losses(t)
    train_unbatched!(g, p, _chn, X, opt, t)
  else
    train_unbatched!(g, p, _chn, X, opt, 10_000)
  end
end


# @inline function batch_size(layers::Tuple, ::Val{T}) where {T}
#   fl = getfield(layers,1)
#   parameter_free(fl) && return batch_size(Base.tail(layers), Val(T))
# end

@generated function turbo_dense_batch_size(
  indputdim::Integer,
  outputdim::Integer,
  Nd::Integer,
  ::StaticInt{W},
  ::StaticInt{RS},
  ::StaticInt{RC},
  ::StaticInt{CLS},
) where {W,RS,RC,CLS}
  Kk = Static.known(indputdim)
  Mk = Static.known(outputdim)
  Nk = Static.known(Nd)
  M = Mk === nothing ? 1024 : Mk
  K = Kk === nothing ? 1024 : Kk
  N = Nk === nothing ? 1024 : Nk
  mₖ, nₖ = matmul_params(RS, RC, CLS; M, K, N, W)
  StaticInt(nₖ)
end
@inline function batch_size(
  layers::Tuple{L,Vararg},
  argsz::Tuple{I,J},
  ::Val{T},
) where {T,L<:TurboDense,I,J}
  inputdim, N = argsz
  outputdim = first(layers).outputdim
  # id, od = getfield(getfield(layers,1), :dims) # (od × id) * (id x N)
  turbo_dense_batch_size(
    inputdim,
    outputdim,
    N,
    VectorizationBase.pick_vector_width(T),
    VectorizationBase.register_size(),
    VectorizationBase.register_count(),
    VectorizationBase.cache_linesize(),
  )
end
@inline batch_size(layers::Tuple{L,Vararg}, argsz::Tuple, ::Val{T}) where {L,T} =
  batch_size(Base.tail(layers), argsz, Val(T))
@inline batch_size(::Tuple{}, ::Tuple, ::Val{T}) where {T} = Static(18)

@inline view_slice_last(X::AbstractArray{<:Any,1}, r) = view(X, r)
@inline view_slice_last(X::AbstractArray{<:Any,2}, r) = view(X, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,3}, r) = view(X, :, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,4}, r) = view(X, :, :, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,5}, r) = view(X, :, :, :, :, r)
function train_batched!(g, p, _chn::Chain, X, opt::AbstractOptimizer, iters)
  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  pen = getpenalty(_chn)
  @unpack layers, memory = chn
  optoff = optmemsize(opt, p)
  sx = ArrayInterface.size(pX)
  resize_memory!(layers, memory, pX, optoff)
  optbuffer, pm = optmemory(opt, p, pointer(memory))
  N = sx[end]
  N_bs =
    batch_size(layers, chain_input_dims(chn, sx), Val(promote_type(eltype(p), eltype(X))))
  d, r = divrem(N, N_bs)
  Ssize = (Base.front(sx)..., N_bs)
  Ssize_rem = (Base.front(sx)..., r)
  GC.@preserve p g memory X begin
    pg = pointer(g)
    pp = pointer(p)
    for _ ∈ 1:iters
      doff = 0
      for d = 1:d
        Xd = view_slice_last(pX, doff+1:doff+N_bs)
        Xp = PtrArray(stridedpointer(Xd), Ssize, StrideArraysCore.val_dense_dims(Xd))
        chain_valgrad_entry!(pg, Xp, layers, pp, pm)
        apply_penalty!(g, pen, p, sx)
        update!(opt, optbuffer, p, g)
        doff += N_bs
      end
      if r ≠ 0
        Xd = view_slice_last(
          pX,
          doff+1:ArrayInterface.static_last(ArrayInterface.axes(X)[end]),
        )
        Xp = PtrArray(stridedpointer(Xd), Ssize_rem, StrideArraysCore.val_dense_dims(Xd))
        chain_valgrad_entry!(pg, Xp, layers, pp, pm)
        apply_penalty!(g, pen, p, sx)
        update!(opt, optbuffer, p, g)
      end
    end
  end
  p
end
train_batched(chn::Chain) = train_batched!(init_params(chn), chn)

_isstochastic(::Tuple{}) = false
function _isstochastic(x::Tuple{T,Vararg}) where {T}
  isstochastic(getfield(x, 1)) ? true : _isstochastic(Base.tail(x))
end

isstochastic(chn::Chain) = _isstochastic(getfield(getchain(chn), :layers))

function train!(g, p, chn::Chain, X, opt::AbstractOptimizer, iters)
  if isstochastic(chn)
    train_unbatched!(g, p, chn, X, opt, iters)
  else
    train_batched!(g, p, chn, X, opt, iters)
  end
end
# train(chn::Chain) = train!(init_params(chn), chn)

for t ∈ [:train, :train_batched, :train_unbatched]
  t! = Symbol(t, :!)
  @eval function $t(chn::Chain, X, opt, iters)
    p = init_params(chn)
    g = similar(p)
    $t!(g, p, chn, X, opt, iters)
  end
end
