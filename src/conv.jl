

# conv functions

# 1d convolution
function convlayer!(_out::AbstractArray{<:Any,2}, _img::AbstractArray{<:Any,2}, _kern::AbstractArray{<:Any,3})
  out = StrideAraysCore.zero_offsets(_out)
  img = StrideAraysCore.zero_offsets(_img)
  kern = StrideAraysCore.zero_offsets(_kern)
  
  @turbo for j₁ ∈ axes(out,1), o ∈ axes(kern,3)
    s = zero(eltype(out))
    for k₁ ∈ axes(kern,1), i ∈ axes(kern,2)
      s += img[j₁ + k₁, i] * kern[k₁, i, o]
    end
    out[j₁, o] = s
  end
end
function convlayer!(_out::AbstractArray{<:Any,3}, _img::AbstractArray{<:Any,3}, _kern::AbstractArray{<:Any,3})
  out = StrideAraysCore.zero_offsets(_out)
  img = StrideAraysCore.zero_offsets(_img)
  kern = StrideAraysCore.zero_offsets(_kern)
  
  for d ∈ axes(out,4)
    convlayer!(view(out,:,:,d), view(img,:,:,d), kern)
  end
end

# 2d convolution
function convlayer!(_out::AbstractArray{<:Any,3}, _img::AbstractArray{<:Any,3}, _kern::AbstractArray{<:Any,4})
  out = StrideAraysCore.zero_offsets(_out)
  img = StrideAraysCore.zero_offsets(_img)
  kern = StrideAraysCore.zero_offsets(_kern)
  @turbo for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), o ∈ axes(kern,4)
    s = zero(eltype(out))
    for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), i ∈ axes(kern,3)
      s += img[j₁ + k₁, j₂ + k₂, i] * kern[k₁, k₂, i, o]
    end
    out[j₁, j₂, o] = s
  end
  out
end
function convlayer!(_out::AbstractArray{<:Any,4}, _img::AbstractArray{<:Any,4}, _kern::AbstractArray{<:Any,4})
  out = StrideAraysCore.zero_offsets(_out)
  img = StrideAraysCore.zero_offsets(_img)
  kern = StrideAraysCore.zero_offsets(_kern)
  for d ∈ axes(out,4)
    convlayer!(view(out,:,:,:,d), view(img,:,:,:,d), kern)
  end
  out
end

function convlayer!(_out::AbstractArray{<:Any,4}, _img::AbstractArray{<:Any,4}, _kern::AbstractArray{<:Any,5})
  out = StrideAraysCore.zero_offsets(_out)
  img = StrideAraysCore.zero_offsets(_img)
  kern = StrideAraysCore.zero_offsets(_kern)
  @turbo for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), j₃ ∈ axes(out,3), o ∈ axes(kern,5)
    s = zero(eltype(out))
    for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), k₃ ∈ axes(kern,3), i ∈ axes(kern,4)
      s += img[j₁ + k₁, j₂ + k₂, j₃ + k₃ - 1, i] * kern[k₁, k₂, k₃, i, o]
    end
    out[j₁, j₂, j₃, o] = s
  end
  out
end
function convlayer!(_out::AbstractArray{<:Any,5}, _img::AbstractArray{<:Any,5}, _kern::AbstractArray{<:Any,5})
  out = StrideAraysCore.zero_offsets(_out)
  img = StrideAraysCore.zero_offsets(_img)
  kern = StrideAraysCore.zero_offsets(_kern)
  for d ∈ axes(out,4)
    convlayer!(view(out,:,:,:,:,d), view(img,:,:,:,:,d), kern)
  end
  out
end

function convlayeradjkern!(_kernadj::AbstractArray{<:Any,3}, _img::AbstractArray{<:Any,2}, _outadj::AbstractArray{<:Any,2})
  outadj = StrideAraysCore.zero_offsets(_outadj)
  img = StrideAraysCore.zero_offsets(_img)
  kernadj = StrideAraysCore.zero_offsets(_kernadj)
  @turbo for k₁ ∈ axes(kernadj,1), i ∈ axes(kernadj,2), o ∈ axes(kernadj,3)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1)
      s += img[j₁ + k₁, i] * outadj[j₁ o]
    end
    kernadj[k₁, i, o] = s
  end
  kernadj
end
function convlayeradjkern!(_kernadj::AbstractArray{<:Any,3}, _img::AbstractArray{<:Any,3}, _outadj::AbstractArray{<:Any,3})
  outadj = StrideAraysCore.zero_offsets(_outadj)
  img = StrideAraysCore.zero_offsets(_img)
  kernadj = StrideAraysCore.zero_offsets(_kernadj)
  @turbo for k₁ ∈ axes(kernadj,1), i ∈ axes(kernadj,2), o ∈ axes(kernadj,3)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1), d ∈ axes(outadj,3)
      s += img[j₁ + k₁, i, d] * outadj[j₁ o, d]
    end
    kernadj[k₁, i, o] = s
  end
  kernadj
end
function convlayeradjkern!(_kernadj::AbstractArray{<:Any,4}, _img::AbstractArray{<:Any,3}, _outadj::AbstractArray{<:Any,3})
  outadj = StrideAraysCore.zero_offsets(_outadj)
  img = StrideAraysCore.zero_offsets(_img)
  kernadj = StrideAraysCore.zero_offsets(_kernadj)
  @turbo for k₁ ∈ axes(kernadj,1), k₂ ∈ axes(kernadj,2), i ∈ axes(kernadj,3), o ∈ axes(kernadj,4)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1), j₂ ∈ axes(outadj,2)
      s += img[j₁ + k₁, j₂ + k₂, i] * outadj[j₁, j₂, o]
    end
    kernadj[k₁, k₂, i, o] = s
  end
  kernadj
end
function convlayeradjkern!(_kernadj::AbstractArray{<:Any,4}, _img::AbstractArray{<:Any,4}, _outadj::AbstractArray{<:Any,4})
  outadj = StrideAraysCore.zero_offsets(_outadj)
  img = StrideAraysCore.zero_offsets(_img)
  kernadj = StrideAraysCore.zero_offsets(_kernadj)
  @turbo for k₁ ∈ axes(kernadj,1), k₂ ∈ axes(kernadj,2), i ∈ axes(kernadj,3), o ∈ axes(kernadj,4)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1), j₂ ∈ axes(outadj,2), d ∈ axes(outadj,4)
      s += img[j₁ + k₁, j₂ + k₂, i, d] * outadj[j₁, j₂, o, d]
    end
    kernadj[k₁, k₂, i, o] = s
  end
  kernadj
end

function convlayeradjkern!(_kernadj::AbstractArray{<:Any,5}, _img::AbstractArray{<:Any,4}, _outadj::AbstractArray{<:Any,4})
  outadj = StrideAraysCore.zero_offsets(_outadj)
  img = StrideAraysCore.zero_offsets(_img)
  kernadj = StrideAraysCore.zero_offsets(_kernadj)
  @turbo for k₁ ∈ axes(kernadj,1), k₂ ∈ axes(kernadj,2), k₃ ∈ axes(kernadj,3), i ∈ axes(kernadj,4), o ∈ axes(kernadj,5)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1), j₂ ∈ axes(outadj,2), j₃ ∈ axes(outadj,3)
      s += img[j₁ + k₁, j₂ + k₂, j₃ + k₃, i] * outadj[j₁, j₂, j₃, o]
    end
    kernadj[k₁, k₂, k₃, i, o] = s
  end
  kernadj
end
function convlayeradjkern!(_kernadj::AbstractArray{<:Any,5}, _img::AbstractArray{<:Any,5}, _outadj::AbstractArray{<:Any,5})
  outadj = StrideAraysCore.zero_offsets(_outadj)
  img = StrideAraysCore.zero_offsets(_img)
  kernadj = StrideAraysCore.zero_offsets(_kernadj)
  @turbo for k₁ ∈ axes(kernadj,1), k₂ ∈ axes(kernadj,2), k₃ ∈ axes(kernadj,3), i ∈ axes(kernadj,4), o ∈ axes(kernadj,5)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1), j₂ ∈ axes(outadj,2), j₃ ∈ axes(outadj,3), d ∈ axes(outadj,5)
      s += img[j₁ + k₁, j₂ + k₂, j₃ + k₃, i, d] * outadj[j₁, j₂, j₃, o, d]
    end
    kernadj[k₁, k₂, k₃, i, o] = s
  end
  kernadj
end

# outadj is padded??


function convlayeradjimg!(
  _imgadj::AbstractArray{<:Any,2},
  _kern::AbstractArray{T,3},
  _outadj::AbstractArray{<:Any,2}
) where {T}
  outadj = StrideAraysCore.zero_offsets(_outadj)
  imgadj = StrideAraysCore.zero_offsets(_imgadj)
  kern = StrideAraysCore.zero_offsets(_kern)
  I0 = size(imgadj,static(1))
  K0, K2, K3 = size(kern)
  J0 = I0 - K0 + static(1)
  for j0 = 0:I0-1, i = 0:K2-1
    s = zero(T)
    for k0 = 0:K0-1, o = 0:K3-1
      ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
      oa = ib0 ? outadj[j0 - k0, o] : zero(T)
      s += kern[k0,i,o] * oa
    end
    imgadj[j0, i] = s
  end
  imgadj
end
function convlayeradjimg!(
  _imgadj::AbstractArray{<:Any,3},
  _kern::AbstractArray{T,3},
  _outadj::AbstractArray{<:Any,3}
) where {T}
  outadj = StrideAraysCore.zero_offsets(_outadj)
  imgadj = StrideAraysCore.zero_offsets(_imgadj)
  kern = StrideAraysCore.zero_offsets(_kern)
  I0, _, I3 = size(imgadj)
  K0, K2, K3 = size(kern)
  J0 = I0 - K0 + static(1)
  for d = 0:I3-1
    for j0 = 0:I0-1, i = 0:K2-1
      s = zero(T)
      for k0 = 0:K0-1, o = 0:K3-1
        ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
        oa = ib0 ? outadj[j0 - k0, o, d] : zero(T)
        s += kern[k0,i,o] * oa
      end
      imgadj[j0, i, d] = s
    end
  end
  imgadj
end
function convlayeradjimg!(
  _imgadj::AbstractArray{<:Any,3},
  _kern::AbstractArray{T,4},
  _outadj::AbstractArray{<:Any,3}
) where {T}
  outadj = StrideAraysCore.zero_offsets(_outadj)
  imgadj = StrideAraysCore.zero_offsets(_imgadj)
  kern = StrideAraysCore.zero_offsets(_kern)
  I0, I1, _ = size(imgadj)
  K0, K1, K2, K3 = size(kern)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  for j0 = 0:I0-1, j1 = 0:I1-1, i = 0:K2-1
    s = zero(T)
    for k0 = 0:K0-1, k1 = 0:K1-1, o = 0:K3-1
      ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
      ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
      oa = (ib0 & ib1) ? outadj[j0 - k0, j1 - k1, o] : zero(T)
      s += kern[k0,k1,i,o] * oa
      end
    imgadj[j0, j1, i] = s
  end
  imgadj
end
function convlayeradjimg!(
  _imgadj::AbstractArray{<:Any,4},
  _kern::AbstractArray{T,4},
  _outadj::AbstractArray{<:Any,4}
) where {T}
  outadj = StrideAraysCore.zero_offsets(_outadj)
  imgadj = StrideAraysCore.zero_offsets(_imgadj)
  kern = StrideAraysCore.zero_offsets(_kern)
  I0, I1, _, I3 = size(imgadj)
  K0, K1, K2, K3 = size(kern)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  for d = 0:I3-1
    for j0 = 0:I0-1, j1 = 0:I1-1, i = 0:K2-1
      s = zero(T)
      for k0 = 0:K0-1, k1 = 0:K1-1, o = 0:K3-1
        ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
        ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
        oa = (ib0 & ib1) ? outadj[j0 - k0, j1 - k1, o, d] : zero(T)
        s += kern[k0,k1,i,o] * oa
      end
      imgadj[j0, j1, i, d] = s
    end
  end
  imgadj
end

#=
# This form is not supported by LoopVectorization:
function convlayeradjimg2!(
  _imgadj,
  _kern::AbstractArray{T,4},
  _outadj
) where {T}
  imgadj = OffsetArray(_imgadj, OffsetArrays.Origin(0))
  kern = OffsetArray(_kern, OffsetArrays.Origin(0))
  outadj = OffsetArray(_outadj, OffsetArrays.Origin(0))
  I0, I1, _, I3 = size(imgadj)
  K0, K1, K2, K3 = size(kern)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  # I0-1 = J0 + K0 - 2
  for d = 0:I3-1
    for j0 = 0:I0-1, j1 = 0:I1-1, i = 0:K2-1
      s = zero(T)
      # for k0 = max(0,j0+1-K0)
      for k0 = max(0,j0-(J0-1)):min(j0, K0-1),
        k1 = max(0,j1-(J1-1)):min(j1, K1-1),
        o = 0:K3-1
        s += kern[K0-1-k0,K1-1-k1,i,o] * outadj[j0 - k0, j1 - k1, o, d]
      end
      imgadj[j0, j1, i, d] = s
    end
  end
  imgadj
end
=#

# generated because `@turbo` prefers literals in indexing expressions
function convlayeradjimg!(
  _imgadj::AbstractArray{<:Any,4},
  _kern::AbstractArray{T,5},
  _outadj::AbstractArray{<:Any,4}
) where {T}
  outadj = StrideAraysCore.zero_offsets(_outadj)
  imgadj = StrideAraysCore.zero_offsets(_imgadj)
  kern = StrideAraysCore.zero_offsets(_kern)
  I0, I1, I2, _ = size(imgadj)
  K0, K1, K2, K3, K4 = size(kern)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  J2 = I2 - K2 + static(1)
  for j0 = 0:I0-1, j1 = 0:I1-1, j2 = 0:I2-1, i = 0:K3-1
    s = zero(T)
    for k0 = 0:K0-1, k1 = 0:K1-1, k2 = 0:K2-1, o = 0:K4-1
      ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
      ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
      ib2 = (j2 - k2 >= 0) & (j2 - k2 < J2)
      oa = (ib0 & ib1 & ib2) ? outadj[j0 - k0, j1 - k1, j2 - k2, o] : zero(T)
      s += kern[k0,k1,k2,i,o] * oa
      end
    imgadj[j0, j1, j2, i] = s
  end
  imgadj
end
function convlayeradjimg!(
  _imgadj::AbstractArray{<:Any,5},
  _kern::AbstractArray{T,5},
  _outadj::AbstractArray{<:Any,5}
) where {T}
  outadj = StrideAraysCore.zero_offsets(_outadj)
  imgadj = StrideAraysCore.zero_offsets(_imgadj)
  kern = StrideAraysCore.zero_offsets(_kern)
  I0, I1, I2, _, I4 = size(imgadj)
  K0, K1, K2, K3, K4 = size(kern)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  J2 = I2 - K2 + static(1)
  for d =  0:I4-1
    for j0 = 0:I0-1, j1 = 0:I1-1, j2 = 0:I2-1, i = 0:K3-1
      s = zero(T)
      for k0 = 0:K0-1, k1 = 0:K1-1, k2 = 0:K2-1, o = 0:K4-1
        ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
        ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
        ib2 = (j2 - k2 >= 0) & (j2 - k2 < J2)
        oa = (ib0 & ib1 & ib2) ? outadj[j0 - k0, j1 - k1, j2 - k2, o, d] : zero(T)
        s += kern[k0,k1,k2,i,o] * oa
      end
      imgadj[j0, j1, j2, i, d] = s
    end
  end
  imgadj
end

struct Conv{D<:Tuple{Vararg{Integer}}}
  dim::D
end

dimsum(c::Conv) = ArrayInterface.reduce_tup(+,c.dim)
dimprod(c::Conv) = ArrayInterface.reduce_tup(*,c.dim)

@inline bsub(::Tuple{}, ::Number) = ()
@inline bsub(x::Tuple{T}, y::Number) where {T} = (only(x) - y,)
@inline bsub(x::Tuple{T0,T1,Vararg}, y::Number) where {T0,T1} = (only(x) - y, bsub(Base.tail(x), y)...)

function getoutputdim(c::Conv{D}, inputdim::Tuple{Vararg{Integer,N0}}) where {N0,N1,D<:Tuple{Vararg{Integer,N1}}}
  @assert N0 + 1 == N1
  cdim0 = c.dim
  out = last(c.dim)

  outdim = bsub(map(+, Base.front(Base.front(inputdim)), Base.front(cdim0)), static(1))
  (outdim..., out)
end
function getoutputdim(c::Conv{D}, inputdim::Tuple{Vararg{Integer,N}}) where {N,D<:Tuple{Vararg{Integer,N}}}
  (getoutputdim(c, Base.front(inputdim))..., last(inputdim))
end

function numparam(c::Conv, inputdim::Tuple{Vararg{Integer}})
  dimprod(c.dim), getoutputdim(c, inputdim)
end

function getparams(c::Conv, p::Ptr{T}) where {T}
  cdim = c.dim
  PtrArray(p, cdim), p + sizeof(T) * prod(cdim)
end

function init_params!(c::Conv, p, inputdim)
  K, p = getparams(c, p)
  gn = Base.FastMath.sqrt_fast(eltype(K)((length(c.dim)-2)/dimsum(c)))
  randn!(local_rng(), K, static(0), static(0), gn)
  return p, getoutputdim(c, inputdim)
end

function alloc_return(c::Conv, inputdim, p)
  outputdim = getoutputdim(c, inputdim)
  R = PtrArray(p, outputdim)
  R, p + align(sizeof(eltype(R))*length(R))
end

function valgrad_layer!(pg::Ptr{T}, c::Conv, img, p::Ptr{T}, pu::Ptr{UInt8}) where {T}
  R, pu3 = alloc_return(c, size(img), Ptr{T}(pu))
  K, p2 = getparams(c, p)
  convlayer!(R, img, K)
  pg + dimprod(c)*sizeof(T), R, p2, Ptr{UInt8}(pu3)
end
function pullback!(pg::Ptr{T}, c::Conv, C̄, img, p::Ptr{T}, _::Ptr{UInt8}, pu2::Ptr{UInt8}) where {T}
  _pullback!(pg, c, C̄, img, p)
  return img, pu2
end
function _pullback!(pg::Ptr{T}, c::Conv, C̄, img, p::Ptr{T}) where {T}
  _pullback_param!(pg, c, C̄, img)
  _pullback_img!(c, C̄, img, p)
  return
end
function _pullback_img!(c::Conv, C̄, img, p::Ptr{T}) where {T}
  K, _  = getparams(c, p)
  # overwrite img
  convlayeradjimg!(img, K, C̄)
  return
end
function pullback_param!(pg::Ptr{T}, c::Conv, C̄, img, ::Ptr{T}, ::Ptr{UInt8}) where {T}
    _pullback_param!(pg, c, C̄, img)
end
function _pullback_param!(pg::Ptr{T}, c::Conv, C̄, img) where {T}
  # get ∂K
  ∂K, _ = getparams(c, pg)
  convlayeradjkern!(∂K, img, C̄)
  return
end

