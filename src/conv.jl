

function convlayer!(out::AbstractArray{<:Any,4}, img, kern)
  for d ∈ axes(out,4)
    for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), o ∈ axes(kern,4)
      s = zero(eltype(out))
      for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), i ∈ axes(kern,3)
        s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * kern[size(kern,1)+1-k₁, size(kern,2)+1-k₂, i, o]
      end
      out[j₁, j₂, o, d] = s
    end
  end
  out
end

function convlayer!(out::AbstractArray{<:Any,5}, img, kern)
  for d ∈ axes(out,4)
    @turbo for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), j₃ ∈ axes(out,3), o ∈ axes(kern,5)
      s = zero(eltype(out))
      for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), k₃ ∈ axes(kern,3), i ∈ axes(kern,4)
        s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, j₃ + k₃ - 1, i, d] * kern[k₁, k₂, k₃, i, o]
      end
      out[j₁, j₂, j₃, o, d] = s
    end
  end
  out
end

function convlayeradjkern!(kernadj::AbstractArray{<:Any,4}, img, outadj)
  @turbo for k₁ ∈ axes(kernadj,1), k₂ ∈ axes(kernadj,2), i ∈ axes(kernadj,3), o ∈ axes(kernadj,4)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1), j₂ ∈ axes(outadj,2), d ∈ axes(outadj,4)
      s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * outadj[j₁, j₂, o, d]
    end
    kernadj[k₁, k₂, i, o] = s
  end
  kernadj
end

function convlayeradjkern!(kernadj::AbstractArray{<:Any,5}, img, outadj)
  @turbo for k₁ ∈ axes(kernadj,1), k₂ ∈ axes(kernadj,2), k₃ ∈ axes(kernadj,3), i ∈ axes(kernadj,4), o ∈ axes(kernadj,5)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1), j₂ ∈ axes(outadj,2), j₃ ∈ axes(outadj,3), d ∈ axes(outadj,5)
      s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, j₃ + k₃ - 1, i, d] * outadj[j₁, j₂, j₃, o, d]
    end
    kernadj[k₁, k₂, k₃, i, o] = s
  end
  kernadj
end

# outadj is padded??

# generated because `@turbo` prefers literals in indexing expressions
@generated function convlayeradjimg!(imgadj, kern::AbstractStrideArray{Tuple{StaticInt{K₁},StaticInt{K₂},StaticInt{I},StaticInt{O}},T,4}, outadj) where {K₁,K₂,I,O,T}
  quote
    for d ∈ axes(outadj,4)
      @turbo for j₁ ∈ axes(imgadj,1), j₂ ∈ axes(imgadj,2), i ∈ axes(kern,3)
        s = zero($T)
        for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), o ∈ axes(kern,4)
          s += kern[k₁, k₂, i, o] * outadj[j₁ - k₁ + $K₁, j₂ - k₂ + $K₂, o, d]
        end
        imgadj[j₁, j₂, i, d] = s
      end
    end
    imgadj
  end
end
using OffsetArrays
function convlayeradjimg!(
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
  for d = 0:I3-1
    for j0 = 0:I0-1, j1 = 0:I1-1, i = 0:K2-1
      s = zero(T)
      for k0 = 0:K0-1, k1 = 0:K1-1, o = 0:K3-1
        ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
        ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
        oa = (ib0 & ib1) ? outadj[j0 - k0, j1 - k1, o, d] : zero(T)
        s += kern[K0-1-k0,K1-1-k1,i,o] * oa
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
  _imgadj,
  _kern::AbstractArray{T,5},
  _outadj
) where {T}
  imgadj = OffsetArray(_imgadj, OffsetArrays.Origin(0))
  kern = OffsetArray(_kern, OffsetArrays.Origin(0))
  outadj = OffsetArray(_outadj, OffsetArrays.Origin(0))
  I0, I1, I2, _, I4 = size(imgadj)
  K0, K1, K2, K3, K4 = size(kern)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  J2 = I2 - K2 + static(1)
  for d = 0:I3-1
    for j0 = 0:I0-1, j1 = 0:I1-1, j2 = 0:I2-1, i = 0:K3-1
      s = zero(T)
      for k0 = 0:K0-1, k1 = 0:K1-1, k2 = 0:K2-1, o = 0:K4-1
        ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
        ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
        ib2 = (j2 - k2 >= 0) & (j2 - k2 < J2)
        oa = (ib0 & ib1 & ib2) ? outadj[j0 - k0, j1 - k1, j2 - k2, o, d] : zero(T)
        s += kern[K0-1-k0,K1-1-k1,K2-1-k2,i,o] * oa
      end
      imgadj[j0, j1, i, d] = s
    end
  end
  imgadj
end

#=
@generated function convlayeradjimg!(imgadj, kern::AbstractStrideArray{Tuple{StaticInt{K₁},StaticInt{K₂},StaticInt{K₃},StaticInt{I},StaticInt{O}},T,5}, outadj) where {K₁,K₂,I,O,T}
  quote
    for d ∈ axes(outadj,5)
      for i ∈ axes(kern,4)
        @turbo for j₁ ∈ axes(imgadj,1), j₂ ∈ axes(imgadj,2), j₃ ∈ axes(imgadj,3)
          s = zero($T)
          for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), k₃ ∈ axes(kern,3), o ∈ axes(kern,5)
            s += kern[k₁, k₂, k₃, i, o] * outadj[j₁ - k₁ + $K₁, j₂ - k₂ + $K₂, j₃ - k₃ + $K₃, o, d]
          end
          imgadj[j₁, j₂, j₃, i, d] = s
        end
      end
    end
    imgadj
  end
end
=#
