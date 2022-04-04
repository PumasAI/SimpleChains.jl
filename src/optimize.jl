
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


function update!(g::AbstractVector, opt, Xp, layers, pen, sx, p, pm, optbuffer, _)
  chain_valgrad_entry!(pointer(g), Xp, layers, pointer(p), pm)
  apply_penalty!(g, pen, p, sx)
  update!(opt, optbuffer, p, g)
end
function chain_valgrad_thread!((g, Xp, layers, p, pm), start, stop, mpt)
  batchsize = size(Xp, ndims(Xp))
  off = start - 1
  nt = size(g, static(2))
  goff = stride(g, 2) * sizeof(eltype(g)) * off
  moff = mpt * off
  f = ((off * batchsize) ÷ nt) + 1
  l = (stop * batchsize) ÷ nt
  Xpv = view_slice_last(Xp, f:l)
  newlayers = (Base.front(layers)..., last(layers)[f:l])
  chain_valgrad_entry!(pointer(g) + goff, Xpv, newlayers, pointer(p), pm + moff)
end
function update!(g::AbstractMatrix, opt, Xp, layers, pen, sx, p, pm, optbuffer, mpt)
  nthread = size(g, static(2))
  Polyester.batch(chain_valgrad_thread!, (nthread, nthread), g, Xp, layers, p, pm, mpt)
  @turbo for t = 2:nthread, i in axes(g, 1)
    g[i, 1] += g[i, t]
  end
  apply_penalty!(g, pen, p, sx)
  GC.@preserve gpb update!(opt, optbuffer, p, PtrArray(pointer(g), (length(p),)))
end
function shuffle_chain_valgrad_thread!(
  (g, Xp, layers, p, pm, mpt, perm, pstart, pstop),
  start,
  stop,
)
  # will work over subrange of pstart+1:pstop
  # it is divided into nthread parts...
  subrangelen = pstop - pstart
  subrangelen > 0 || return
  numthread = size(g, static(2))
  batchsize, r = divrem(subrangelen, numthread)
  off = start - 1
  goff = stride(g, 2) * sizeof(eltype(g)) * off
  pm += mpt * off

  fm1 = off * batchsize + pstart + min(r, off)
  lastdim = batchsize + (start <= r)
  l = fm1 + lastdim

  loss = last(layers)
  tgt = target(loss)
  # @show size(tgt)
  tgtpb = preserve_buffer(tgt)
  Xpb = preserve_buffer(Xp)
  Xsz = Base.front(size(Xp))
  eltx = eltype(Xp)
  eltgt = eltype(tgt)
  szeltx = sizeof(eltx)
  szeltgt = sizeof(eltgt)

  Xtmp = PtrArray(Ptr{eltx}(pm), (Xsz..., lastdim))
  Xlen = tsprod(Xsz)
  pXtmp = pointer(Xtmp)
  pm += align(sizeof(eltgt) * Xlen * lastdim)
  tgtsz = Base.front(size(tgt))
  tgttmp = PtrArray(Ptr{eltgt}(pm), (tgtsz..., lastdim))
  ptgttmp = pointer(tgttmp)
  tgtlen = tsprod(tgtsz)
  pm += align(szeltgt * tgtlen * lastdim)
  pX = pointer(Xp)
  ptgt = pointer(tgt)
  GC.@preserve tgtpb Xpb begin
    for i = fm1:l-1
      j = perm[i+1]
      # @show i, j
      @simd ivdep for k = 0:Int(tgtlen)-1
        x = unsafe_load((ptgt + (tgtlen * szeltgt) * j) + k * szeltgt)
        unsafe_store!(ptgttmp + k * szeltgt, x)
      end
      # Base.unsafe_copyto!(ptgttmp, ptgt + tgtlen*szeltgt*j, Int(tgtlen))
      Base.unsafe_copyto!(pXtmp, pX + Xlen * szeltx * j, Int(Xlen))
      ptgttmp += Int(tgtlen) * szeltgt
      pXtmp += Int(Xlen) * szeltx
    end
  end
  # @show 1+fm1:l batchsize Xtmp tgttmp tgtlen Xlen lastdim
  newlayers = (Base.front(layers)..., loss(tgttmp))
  chain_valgrad_entry!(pointer(g) + goff, Xtmp, newlayers, pointer(p), pm)
  return nothing
end
function shuffle_update!(
  g::AbstractMatrix,
  opt,
  Xp,
  layers,
  pen,
  sx,
  p,
  pm,
  optbuffer,
  mpt,
  perm,
  pstart,
  pstop,
)
  nthread = size(g, static(2))
  Polyester.batch(
    shuffle_chain_valgrad_thread!,
    (nthread, nthread),
    g,
    Xp,
    layers,
    p,
    pm,
    mpt,
    perm,
    pstart,
    pstop,
  )
  @turbo for t = 2:nthread, i in axes(g, 1)
    g[i, 1] += g[i, t]
  end
  apply_penalty!(g, pen, p, sx)
  gpb = preserve_buffer(g)
  GC.@preserve gpb update!(opt, optbuffer, p, PtrArray(pointer(g), (length(p),)))
end
function shuffle_update!(
  g::AbstractVector,
  opt,
  Xp,
  layers,
  pen,
  sx,
  p,
  pm,
  optbuffer,
  mpt,
  perm,
  pstart,
  pstop,
)
  shuffle_chain_valgrad_thread!(
    (g, Xp, layers, p, pm, mpt, perm, pstart, pstop),
    static(1),
    static(1),
  )
  apply_penalty!(g, pen, p, sx)
  update!(opt, optbuffer, p, g)
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
  mpt = resize_memory!(layers, memory, pX, optoff, size(g, static(2)))
  optbuffer, pm = optmemory(opt, p, pointer(memory))
  GC.@preserve p g memory X begin
    for y ∈ t
      layers_y = (fl..., ll(y))
      update!(g, opt, pX, layers_y, pen, sx, p, pm, optbuffer, mpt)
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
  mpt = resize_memory!(layers, memory, pX, optoff, size(g, static(2)))
  optbuffer, pm = optmemory(opt, p, pointer(memory))
  GC.@preserve p g memory X begin
    for _ ∈ 1:iters
      update!(g, opt, pX, layers, pen, sx, p, pm, optbuffer, mpt)
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
    register_size(),
    register_count(),
    cache_linesize(),
  )
end
@inline function batch_size(layers::Tuple{L,Vararg}, argsz::Tuple, ::Val{T}) where {L,T}
  _, argsz2 = layer_output_size(Val{T}(), getfield(layers, 1), argsz)
  batch_size(Base.tail(layers), argsz2, Val(T))
end
@inline batch_size(::Tuple{}, ::Tuple, ::Val{T}) where {T} = static(18)


@inline view_slice_last(X::AbstractArray{<:Any,1}, r) = view(X, r)
@inline view_slice_last(X::AbstractArray{<:Any,2}, r) = view(X, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,3}, r) = view(X, :, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,4}, r) = view(X, :, :, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,5}, r) = view(X, :, :, :, :, r)
function train_batched!(
  g,
  p,
  _chn::Chain,
  X,
  opt::AbstractOptimizer,
  iters;
  batchsize = nothing,
)
  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  pen = getpenalty(_chn)
  @unpack layers, memory = chn
  optoff = optmemsize(opt, p)
  sx = chain_input_dims(chn, size(pX))
  N = sx[end]
  # need to shuffle `N`
  tgt = target(chn)
  nthread = size(g, static(2))
  N_bs =
    batchsize === nothing ?
    batch_size(layers, sx, Val(promote_type(eltype(p), eltype(X)))) * nthread :
    batchsize
  tgt_batch_len = tsprod(Base.front(size(tgt))) * N_bs
  X_batch_len = tsprod(Base.front(sx)) * N_bs
  shuffle_per_thread =
    align(sizeof(eltype(tgt)) * tgt_batch_len) + align(sizeof(eltype(X)) * X_batch_len)
  shuffle_off = align(sizeof(Int) * N_bs) + shuffle_per_thread * nthread
  mpt =
    resize_memory!(layers, memory, pX, optoff + shuffle_off, nthread) + shuffle_per_thread
  optbuffer, pm = optmemory(opt, p, pointer(memory))
  perm = PtrArray(Ptr{Int}(pm), (N,))
  for n = 0:N-1
    perm[n+1] = n
  end
  pm += align(sizeof(Int) * N)
  d, r = divrem(N, N_bs)
  d += r != 0
  r = ifelse(r != 0, r, N_bs)
  GC.@preserve p g memory X begin
    iter = 0
    while true
      doff = 0
      while true
        doffnext = doff + N_bs
        batchstop::Int = min(doffnext, N)
        # Xd = view_slice_last(pX, doff+1:batchstop)
        # newlayers = (Base.front(layers)..., last(layers)[doff+1:batchstop])
        # shuffle_update!(g, opt, Xd, newlayers, pen, sx, p, pm, optbuffer, mpt, perm)
        shuffle_update!(
          g,
          opt,
          pX,
          layers,
          pen,
          sx,
          p,
          pm,
          optbuffer,
          mpt,
          perm,
          doff,
          batchstop,
        )
        doff = doffnext
        doff >= N && break
      end
      (iter += 1) < iters || break
      Random.shuffle!(perm)
    end
  end
  p
end

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
