
# conv functions

# 1d convolution
function convlayer!(
  f::F,
  _C::AbstractArray{<:Any,2},
  _A::AbstractArray{<:Any,2},
  _K::AbstractArray{<:Any,3},
  _b::AbstractVector
) where {F}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)

  @turbo warn_check_args = false for j₁ ∈ axes(C, 1), o ∈ axes(K, 3)
    s = zero(eltype(C))
    for k₁ ∈ axes(K, 1), i ∈ axes(K, 2)
      s += A[j₁+k₁, i] * K[k₁, i, o]
    end
    C[j₁, o] = f(s + b[o])
  end
end
function convlayer!(
  f::F,
  _C::AbstractArray{<:Any,3},
  _A::AbstractArray{<:Any,3},
  _K::AbstractArray{<:Any,3},
  _b::AbstractVector
) where {F}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)

  for d ∈ axes(C, 4)
    convlayer!(f, zview(C, :, :, d), zview(A, :, :, d), K, b)
  end
end
function convlayer!(
  f::F,
  _C::AbstractArray{<:Any,3},
  _A::AbstractArray{<:Any,3},
  _K::AbstractArray{<:Any,3},
  _b::AbstractVector,
  _inds::AbstractVector{<:Integer}
) where {F}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  inds = zero_offsets(_inds)
  for d ∈ axes(C, 4)
    convlayer!(f, zview(C, :, :, d), zview(A, :, :, inds[d]), K, b)
  end
end
function convlayer!(
  ∂f::F,
  _∂C::AbstractArray{<:Any,2},
  _C::AbstractArray{<:Any,2},
  _A::AbstractArray{<:Any,2},
  _K::AbstractArray{<:Any,3},
  _b::AbstractVector
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)

  @turbo warn_check_args = false for j₁ ∈ axes(C, 1), o ∈ axes(K, 3)
    s = zero(eltype(C))
    for k₁ ∈ axes(K, 1), i ∈ axes(K, 2)
      s += A[j₁+k₁, i] * K[k₁, i, o]
    end
    Cjo, ∂Cjo = ∂f(s + b[o])
    C[j₁, o] = Cjo
    ∂C[j₁, o] = ∂Cjo
  end
end
function convlayer!(
  f::F,
  _∂C::AbstractArray{<:Any,3},
  _C::AbstractArray{<:Any,3},
  _A::AbstractArray{<:Any,3},
  _K::AbstractArray{<:Any,3},
  _b::AbstractVector
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  for d ∈ axes(C, 4)
    convlayer!(
      f,
      zview(∂C, :, :, d),
      zview(C, :, :, d),
      zview(A, :, :, d),
      K,
      b
    )
  end
end
function convlayer!(
  f::F,
  _∂C::AbstractArray{<:Any,3},
  _C::AbstractArray{<:Any,3},
  _A::AbstractArray{<:Any,3},
  _K::AbstractArray{<:Any,3},
  _b::AbstractVector,
  _inds::AbstractVector{<:Integer}
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  inds = zero_offsets(_inds)
  for d ∈ axes(C, 4)
    convlayer!(
      f,
      zview(∂C, :, :, d),
      zview(C, :, :, d),
      zview(A, :, :, inds[d]),
      K,
      b
    )
  end
end

# 2d convolution
function convlayer!(
  f::F,
  _C::AbstractArray{<:Any,3},
  _A::AbstractArray{<:Any,3},
  _K::AbstractArray{<:Any,4},
  _b::AbstractVector
) where {F}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  @turbo warn_check_args = false for j₁ ∈ axes(C, 1),
    j₂ ∈ axes(C, 2),
    o ∈ axes(K, 4)

    s = zero(eltype(C))
    for k₁ ∈ axes(K, 1), k₂ ∈ axes(K, 2), i ∈ axes(K, 3)
      s += A[j₁+k₁, j₂+k₂, i] * K[k₁, k₂, i, o]
    end
    C[j₁, j₂, o] = f(s + b[o])
  end
end
function convlayer!(
  f::F,
  _C::AbstractArray{<:Any,4},
  _A::AbstractArray{<:Any,4},
  _K::AbstractArray{<:Any,4},
  _b::AbstractVector
) where {F}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  for d ∈ axes(C, 4)
    convlayer!(f, zview(C, :, :, :, d), zview(A, :, :, :, d), K, b)
  end
end
function convlayer!(
  f::F,
  _C::AbstractArray{<:Any,4},
  _A::AbstractArray{<:Any,4},
  _K::AbstractArray{<:Any,4},
  _b::AbstractVector,
  _inds::AbstractVector{<:Integer}
) where {F}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  inds = zero_offsets(_inds)
  for d ∈ axes(C, 4)
    convlayer!(f, zview(C, :, :, :, d), zview(A, :, :, :, inds[d]), K, b)
  end
end
# 3d convolution
function convlayer!(
  f::F,
  _C::AbstractArray{<:Any,4},
  _A::AbstractArray{<:Any,4},
  _K::AbstractArray{<:Any,5},
  _b::AbstractVector
) where {F}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  @turbo warn_check_args = false for j₁ ∈ axes(C, 1),
    j₂ ∈ axes(C, 2),
    j₃ ∈ axes(C, 3),
    o ∈ axes(K, 5)

    s = zero(eltype(C))
    for k₁ ∈ axes(K, 1), k₂ ∈ axes(K, 2), k₃ ∈ axes(K, 3), i ∈ axes(K, 4)
      s += A[j₁+k₁, j₂+k₂, j₃+k₃-1, i] * K[k₁, k₂, k₃, i, o]
    end
    C[j₁, j₂, j₃, o] = f(s + b[o])
  end
end
function convlayer!(
  f::F,
  _C::AbstractArray{<:Any,5},
  _A::AbstractArray{<:Any,5},
  _K::AbstractArray{<:Any,5},
  _b::AbstractVector
) where {F}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  for d ∈ axes(C, 4)
    convlayer!(f, zview(C, :, :, :, :, d), zview(A, :, :, :, :, d), K, b)
  end
end
function convlayer!(
  f::F,
  _C::AbstractArray{<:Any,5},
  _A::AbstractArray{<:Any,5},
  _K::AbstractArray{<:Any,5},
  _b::AbstractVector,
  _inds::AbstractVector{<:Integer}
) where {F}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  inds = zero_offsets(_inds)
  for d ∈ axes(C, 4)
    convlayer!(f, zview(C, :, :, :, :, d), zview(A, :, :, :, :, inds[d]), K, b)
  end
end

# 2d convolution
function convlayer!(
  ∂f::F,
  _∂C::AbstractArray{<:Any,3},
  _C::AbstractArray{<:Any,3},
  _A::AbstractArray{<:Any,3},
  _K::AbstractArray{<:Any,4},
  _b::AbstractVector
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  # FIXME: this seems to be buggy!!! The outer `@turbo` definitely
  # results in worse accuracy after training
  for o ∈ axes(K, 4)
    @turbo warn_check_args = false for j₁ ∈ axes(C, 1), j₂ ∈ axes(C, 2)
      # @turbo for j₁ ∈ axes(C, 1), j₂ ∈ axes(C, 2), o ∈ axes(K, 4)
      # for j₁ ∈ axes(C, 1), j₂ ∈ axes(C, 2), o ∈ axes(K, 4)
      s = zero(eltype(C))
      for k₁ ∈ axes(K, 1), k₂ ∈ axes(K, 2), i ∈ axes(K, 3)
        s += A[j₁+k₁, j₂+k₂, i] * K[k₁, k₂, i, o]
      end
      C[j₁, j₂, o], ∂C[j₁, j₂, o] = ∂f(s + b[o])
    end
  end
  # end
end
function convlayer!(
  f::F,
  _∂C::AbstractArray{<:Any,4},
  _C::AbstractArray{<:Any,4},
  _A::AbstractArray{<:Any,4},
  _K::AbstractArray{<:Any,4},
  _b::AbstractVector
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  for d ∈ axes(C, 4)
    convlayer!(
      f,
      zview(∂C, :, :, :, d),
      zview(C, :, :, :, d),
      zview(A, :, :, :, d),
      K,
      b
    )
  end
end
function convlayer!(
  f::F,
  _∂C::AbstractArray{<:Any,4},
  _C::AbstractArray{<:Any,4},
  _A::AbstractArray{<:Any,4},
  _K::AbstractArray{<:Any,4},
  _b::AbstractVector,
  _inds::AbstractVector{<:Integer}
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  inds = zero_offsets(_inds)
  for d ∈ axes(C, 4)
    convlayer!(
      f,
      zview(∂C, :, :, :, d),
      zview(C, :, :, :, d),
      zview(A, :, :, :, inds[d]),
      K,
      b
    )
  end
end

# 3d convolution
function convlayer!(
  ∂f::F,
  _∂C::AbstractArray{<:Any,4},
  _C::AbstractArray{<:Any,4},
  _A::AbstractArray{<:Any,4},
  _K::AbstractArray{<:Any,5},
  _b::AbstractVector
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  @turbo warn_check_args = false for j₁ ∈ axes(C, 1),
    j₂ ∈ axes(C, 2),
    j₃ ∈ axes(C, 3),
    o ∈ axes(K, 5)

    s = zero(eltype(C))
    for k₁ ∈ axes(K, 1), k₂ ∈ axes(K, 2), k₃ ∈ axes(K, 3), i ∈ axes(K, 4)
      s += A[j₁+k₁, j₂+k₂, j₃+k₃-1, i] * K[k₁, k₂, k₃, i, o]
    end
    c, ∂c = ∂f(s + b[o])
    C[j₁, j₂, j₃, o] = c
    ∂C[j₁, j₂, j₃, o] = ∂c
  end
end
function convlayer!(
  f::F,
  ∂C::AbstractArray{<:Any,5},
  C::AbstractArray{<:Any,5},
  A::AbstractArray{<:Any,5},
  K::AbstractArray{<:Any,5},
  b::AbstractVector
) where {F}
  for d ∈ axes(C, 4)
    convlayer!(
      f,
      zview(∂C, :, :, :, :, d),
      zview(C, :, :, :, :, d),
      zview(A, :, :, :, :, d),
      K,
      b
    )
  end
end
function convlayer!(
  f::F,
  _∂C::AbstractArray{<:Any,5},
  _C::AbstractArray{<:Any,5},
  _A::AbstractArray{<:Any,5},
  _K::AbstractArray{<:Any,5},
  _b::AbstractVector,
  _inds::AbstractVector{<:Integer}
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  K = zero_offsets(_K)
  b = zero_offsets(_b)
  inds = zero_offsets(_inds)
  for d ∈ axes(C, 4)
    convlayer!(
      f,
      zview(∂C, :, :, :, :, d),
      zview(C, :, :, :, :, d),
      zview(A, :, :, :, :, inds[d]),
      K,
      b
    )
  end
end

function convbadjoint!(_badj::AbstractVector, _Cadj::AbstractArray{<:Any,2})
  badj = zero_offsets(_badj)
  Cadj = zero_offsets(_Cadj)
  @turbo warn_check_args = false for o ∈ axes(Cadj, 2)
    s = zero(eltype(badj))
    for j ∈ axes(Cadj, 1)
      s += Cadj[j, o]
    end
    badj[o] = s
  end
end
function convlayeradjK!(
  _Kadj::AbstractArray{<:Any,3},
  _badj,
  _A::AbstractArray{<:Any,2},
  _Cadj::AbstractArray{<:Any,2}
)
  Cadj = zero_offsets(_Cadj)
  A = zero_offsets(_A)
  Kadj = zero_offsets(_Kadj)
  @turbo warn_check_args = false for k₁ ∈ axes(Kadj, 1),
    i ∈ axes(Kadj, 2),
    o ∈ axes(Kadj, 3)

    s = zero(eltype(Kadj))
    for j₁ ∈ axes(Cadj, 1)
      s += A[j₁+k₁, i] * Cadj[j₁, o]
    end
    Kadj[k₁, i, o] = s
  end
  convbadjoint!(_badj, Cadj)
end
function convbadjoint!(_badj::AbstractVector, _Cadj::AbstractArray{<:Any,3})
  badj = zero_offsets(_badj)
  Cadj = zero_offsets(_Cadj)
  @turbo warn_check_args = false for o ∈ axes(Cadj, 2)
    s = zero(eltype(badj))
    for j ∈ axes(Cadj, 1), d ∈ axes(Cadj, 3)
      s += Cadj[j, o, d]
    end
    badj[o] = s
  end
end
function convlayeradjK!(
  _Kadj::AbstractArray{<:Any,3},
  _badj,
  _A::AbstractArray{<:Any,3},
  _Cadj::AbstractArray{<:Any,3}
)
  Cadj = zero_offsets(_Cadj)
  A = zero_offsets(_A)
  Kadj = zero_offsets(_Kadj)
  @turbo warn_check_args = false for k₁ ∈ axes(Kadj, 1),
    i ∈ axes(Kadj, 2),
    o ∈ axes(Kadj, 3)

    s = zero(eltype(Kadj))
    for j₁ ∈ axes(Cadj, 1), d ∈ axes(Cadj, 3)
      s += A[j₁+k₁, i, d] * Cadj[j₁, o, d]
    end
    Kadj[k₁, i, o] = s
  end
  convbadjoint!(_badj, Cadj)
end
function convlayeradjK!(
  _Kadj::AbstractArray{<:Any,4},
  _badj,
  _A::AbstractArray{<:Any,3},
  _Cadj::AbstractArray{<:Any,3}
)
  Cadj = zero_offsets(_Cadj)
  A = zero_offsets(_A)
  Kadj = zero_offsets(_Kadj)
  @turbo warn_check_args = false for k₁ ∈ axes(Kadj, 1),
    k₂ ∈ axes(Kadj, 2),
    i ∈ axes(Kadj, 3),
    o ∈ axes(Kadj, 4)

    s = zero(eltype(Kadj))
    for j₁ ∈ axes(Cadj, 1), j₂ ∈ axes(Cadj, 2)
      s += A[j₁+k₁, j₂+k₂, i] * Cadj[j₁, j₂, o]
    end
    Kadj[k₁, k₂, i, o] = s
  end
  convbadjoint!(_badj, Flatten{2}()(Cadj))
end
function convlayeradjK!(
  _Kadj::AbstractArray{<:Any,4},
  _badj,
  _A::AbstractArray{<:Any,4},
  _Cadj::AbstractArray{<:Any,4}
)
  Cadj = zero_offsets(_Cadj)
  A = zero_offsets(_A)
  Kadj = zero_offsets(_Kadj)
  @turbo warn_check_args = false for k₁ ∈ axes(Kadj, 1),
    k₂ ∈ axes(Kadj, 2),
    i ∈ axes(Kadj, 3),
    o ∈ axes(Kadj, 4)

    s = zero(eltype(Kadj))
    for j₁ ∈ axes(Cadj, 1), j₂ ∈ axes(Cadj, 2), d ∈ axes(Cadj, 4)
      s += A[j₁+k₁, j₂+k₂, i, d] * Cadj[j₁, j₂, o, d]
    end
    Kadj[k₁, k₂, i, o] = s
  end
  convbadjoint!(_badj, Flatten{2}()(Cadj))
end

function convlayeradjK!(
  _Kadj::AbstractArray{<:Any,5},
  _badj,
  _A::AbstractArray{<:Any,4},
  _Cadj::AbstractArray{<:Any,4}
)
  Cadj = zero_offsets(_Cadj)
  A = zero_offsets(_A)
  Kadj = zero_offsets(_Kadj)
  @turbo warn_check_args = false for k₁ ∈ axes(Kadj, 1),
    k₂ ∈ axes(Kadj, 2),
    k₃ ∈ axes(Kadj, 3),
    i ∈ axes(Kadj, 4),
    o ∈ axes(Kadj, 5)

    s = zero(eltype(Kadj))
    for j₁ ∈ axes(Cadj, 1), j₂ ∈ axes(Cadj, 2), j₃ ∈ axes(Cadj, 3)
      s += A[j₁+k₁, j₂+k₂, j₃+k₃, i] * Cadj[j₁, j₂, j₃, o]
    end
    Kadj[k₁, k₂, k₃, i, o] = s
  end
  convbadjoint!(_badj, Flatten{3}()(Cadj))
end
function convlayeradjK!(
  _Kadj::AbstractArray{<:Any,5},
  _badj,
  _A::AbstractArray{<:Any,5},
  _Cadj::AbstractArray{<:Any,5}
)
  Cadj = zero_offsets(_Cadj)
  A = zero_offsets(_A)
  Kadj = zero_offsets(_Kadj)
  @turbo warn_check_args = false for k₁ ∈ axes(Kadj, 1),
    k₂ ∈ axes(Kadj, 2),
    k₃ ∈ axes(Kadj, 3),
    i ∈ axes(Kadj, 4),
    o ∈ axes(Kadj, 5)

    s = zero(eltype(Kadj))
    for j₁ ∈ axes(Cadj, 1),
      j₂ ∈ axes(Cadj, 2),
      j₃ ∈ axes(Cadj, 3),
      d ∈ axes(Cadj, 5)

      s += A[j₁+k₁, j₂+k₂, j₃+k₃, i, d] * Cadj[j₁, j₂, j₃, o, d]
    end
    Kadj[k₁, k₂, k₃, i, o] = s
  end
  convbadjoint!(_badj, Flatten{3}()(Cadj))
end

# Cadj is padded??

function convlayeradjA!(
  _Aadj::AbstractArray{<:Any,2},
  _K::AbstractArray{T,3},
  _Cadj::AbstractArray{<:Any,2}
) where {T}
  Cadj = zero_offsets(_Cadj)
  Aadj = zero_offsets(_Aadj)
  K = zero_offsets(_K)
  I0 = static_size(Aadj, static(1))
  K0, K2, K3 = static_size(K)
  J0 = I0 - K0 + static(1)
  @turbo warn_check_args = false for j0 = 0:I0-1, i = 0:K2-1
    s = zero(T)
    for k0 = 0:K0-1, o = 0:K3-1
      ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
      oa = ib0 ? Cadj[j0-k0, o] : zero(T)
      s += K[k0, i, o] * oa
    end
    Aadj[j0, i] = s
  end
end
function convlayeradjA!(
  _Aadj::AbstractArray{<:Any,3},
  _K::AbstractArray{T,3},
  _Cadj::AbstractArray{<:Any,3}
) where {T}
  Cadj = zero_offsets(_Cadj)
  Aadj = zero_offsets(_Aadj)
  K = zero_offsets(_K)
  I0, _, I3 = static_size(Aadj)
  K0, K2, K3 = static_size(K)
  J0 = I0 - K0 + static(1)
  for d = 0:I3-1
    @turbo warn_check_args = false for j0 = 0:I0-1, i = 0:K2-1
      s = zero(T)
      for k0 = 0:K0-1, o = 0:K3-1
        ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
        oa = ib0 ? Cadj[j0-k0, o, d] : zero(T)
        s += K[k0, i, o] * oa
      end
      Aadj[j0, i, d] = s
    end
  end
end
function convlayeradjA!(
  _Aadj::AbstractArray{<:Any,3},
  _K::AbstractArray{T,4},
  _Cadj::AbstractArray{<:Any,3}
) where {T}
  Cadj = zero_offsets(_Cadj)
  Aadj = zero_offsets(_Aadj)
  K = zero_offsets(_K)
  I0, I1, _ = static_size(Aadj)
  K0, K1, K2, K3 = static_size(K)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  @turbo warn_check_args = false for j0 = 0:I0-1, j1 = 0:I1-1, i = 0:K2-1
    s = zero(T)
    for k0 = 0:K0-1, k1 = 0:K1-1, o = 0:K3-1
      ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
      ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
      oa = (ib0 & ib1) ? Cadj[j0-k0, j1-k1, o] : zero(T)
      s += K[k0, k1, i, o] * oa
    end
    Aadj[j0, j1, i] = s
  end
end
function convlayeradjA!(
  _Aadj::AbstractArray{<:Any,4},
  _K::AbstractArray{T,4},
  _Cadj::AbstractArray{<:Any,4}
) where {T}
  Cadj = zero_offsets(_Cadj)
  Aadj = zero_offsets(_Aadj)
  K = zero_offsets(_K)
  I0, I1, _, I3 = static_size(Aadj)
  K0, K1, K2, K3 = static_size(K)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  for d = 0:I3-1
    @turbo warn_check_args = false unroll = (2, 1) for j0 = 0:I0-1,
      j1 = 0:I1-1,
      i = 0:K2-1

      s = zero(T)
      for k0 = 0:K0-1, k1 = 0:K1-1, o = 0:K3-1
        ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
        ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
        oa = (ib0 & ib1) ? Cadj[j0-k0, j1-k1, o, d] : zero(T)
        s += K[k0, k1, i, o] * oa
      end
      Aadj[j0, j1, i, d] = s
    end
  end
end

#=
# This form is not supported by LoopVectorization:
function convlayeradjA2!(
  _Aadj,
  _K::AbstractArray{T,4},
  _Cadj
) where {T}
  Aadj = OffsetArray(_Aadj, OffsetArrays.Origin(0))
  K = OffsetArray(_K, OffsetArrays.Origin(0))
  Cadj = OffsetArray(_Cadj, OffsetArrays.Origin(0))
  I0, I1, _, I3 = static_size(Aadj)
  K0, K1, K2, K3 = static_size(K)
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
        s += K[K0-1-k0,K1-1-k1,i,o] * Cadj[j0 - k0, j1 - k1, o, d]
      end
      Aadj[j0, j1, i, d] = s
    end
  end
  Aadj
end
=#

# generated because `#= @turbo =#` prefers literals in indexing expressions
function convlayeradjA!(
  _Aadj::AbstractArray{<:Any,4},
  _K::AbstractArray{T,5},
  _Cadj::AbstractArray{<:Any,4}
) where {T}
  Cadj = zero_offsets(_Cadj)
  Aadj = zero_offsets(_Aadj)
  K = zero_offsets(_K)
  I0, I1, I2, _ = static_size(Aadj)
  K0, K1, K2, K3, K4 = static_size(K)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  J2 = I2 - K2 + static(1)
  @turbo warn_check_args = false for j0 = 0:I0-1,
    j1 = 0:I1-1,
    j2 = 0:I2-1,
    i = 0:K3-1

    s = zero(T)
    for k0 = 0:K0-1, k1 = 0:K1-1, k2 = 0:K2-1, o = 0:K4-1
      ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
      ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
      ib2 = (j2 - k2 >= 0) & (j2 - k2 < J2)
      oa = (ib0 & ib1 & ib2) ? Cadj[j0-k0, j1-k1, j2-k2, o] : zero(T)
      s += K[k0, k1, k2, i, o] * oa
    end
    Aadj[j0, j1, j2, i] = s
  end
end
function convlayeradjA!(
  _Aadj::AbstractArray{<:Any,5},
  _K::AbstractArray{T,5},
  _Cadj::AbstractArray{<:Any,5}
) where {T}
  Cadj = zero_offsets(_Cadj)
  Aadj = zero_offsets(_Aadj)
  K = zero_offsets(_K)
  I0, I1, I2, _, I4 = static_size(Aadj)
  K0, K1, K2, K3, K4 = static_size(K)
  J0 = I0 - K0 + static(1)
  J1 = I1 - K1 + static(1)
  J2 = I2 - K2 + static(1)
  for d = 0:I4-1
    @turbo warn_check_args = false for j0 = 0:I0-1,
      j1 = 0:I1-1,
      j2 = 0:I2-1,
      i = 0:K3-1

      s = zero(T)
      for k0 = 0:K0-1, k1 = 0:K1-1, k2 = 0:K2-1, o = 0:K4-1
        ib0 = (j0 - k0 >= 0) & (j0 - k0 < J0)
        ib1 = (j1 - k1 >= 0) & (j1 - k1 < J1)
        ib2 = (j2 - k2 >= 0) & (j2 - k2 < J2)
        oa = (ib0 & ib1 & ib2) ? Cadj[j0-k0, j1-k1, j2-k2, o, d] : zero(T)
        s += K[k0, k1, k2, i, o] * oa
      end
      Aadj[j0, j1, j2, i, d] = s
    end
  end
end

"""
    Conv(activation, dims::Tuple{Vararg{Integer}}, outputdim::Integer)

Performs a convolution with `dims` and maps it to `outputdim` output channels, then
adds a bias (one per `outputdim`) and applies `activation` elementwise.

E.g., `Conv(relu, (5, 5), 16)` performs a `5 × 5` convolution, and maps the input
channels to 16 output channels, before adding a bias and applying `relu`.

Randomly initializing weights using the (Xavier) Glorot uniform distribution.
The bias is zero-initialized.
"""
struct Conv{F,D<:Tuple{Vararg{Integer}},O<:Integer}
  dim::D
  outputdim::O
  f::F
end
function Conv(
  f::F,
  dims::Tuple{Vararg{Integer,K}},
  outputdim::Integer
) where {F,K}
  Conv(map(static, dims), static(outputdim), f)
end
function Conv(dims::Tuple{Vararg{Integer,K}}, outputdim::Integer) where {K}
  Conv(map(static, dims), static(outputdim), identity)
end
fast_fuse(c::Conv) = fast_fuse(getfield(c, :f))

_fused_fun(c, ::True) = getfield(c, :f)
_fused_fun(_, ::False) = identity
fused_fun(c) = _fused_fun(c, fast_fuse(c))

_unfused_fun(_, ::True) = identity
_unfused_fun(c, ::False) = getfield(c, :f)
unfused_fun(c) = Activation(_unfused_fun(c, fast_fuse(c)))

dimsum(c::Conv) = ArrayInterface.reduce_tup(+, c.dim)
dimprod(c::Conv) = tsprod(c.dim)

function Base.show(io::IO, c::Conv)
  print(io, "Conv $(c.dim) mapping to $(c.outputdim)")
  if c.f !== identity
    println(io)
    show(io, Activation(c.f))
  end
end

function getoutputdim(
  c::Conv{F,D},
  inputdim::Tuple{Vararg{Integer,N}}
) where {F,N,D<:Tuple{Vararg{Integer,N}}}
  badd(map(-, inputdim, c.dim), static(1))
end
function _getoutputdim(
  c::Conv{F,D},
  inputdim::Tuple{Vararg{Integer,N}},
  ::Integer
) where {F,N,D<:Tuple{Vararg{Integer,N}}}
  # ignored arg is replaced with c.outputdim
  (getoutputdim(c, inputdim)..., c.outputdim)
end
function _getoutputdim(
  c::Conv{F,D},
  inputdim::Tuple{Vararg{Integer,N0}},
  lastinputdim::Integer
) where {F,N0,N1,D<:Tuple{Vararg{Integer,N1}}}
  # lastinputdim is the batch size
  (getoutputdim(c, Base.front(inputdim))..., c.outputdim, lastinputdim)
end
function getoutputdim(
  c::Conv{F,D},
  inputdim::Tuple{Vararg{Integer,N0}}
) where {F,N0,N1,D<:Tuple{Vararg{Integer,N1}}}
  _getoutputdim(c, Base.front(inputdim), last(inputdim))
end

function _paramdim(
  c::Conv{F,D},
  ::Tuple{Vararg{Integer,N}},
  lastdim::Integer
) where {F,N,D<:Tuple{Vararg{Integer,N}}}
  (c.dim..., lastdim, c.outputdim)
end
function _paramdim(
  c::Conv{F,D},
  inputdim::Tuple{Vararg{Integer,N0}},
  lastdim::Integer
) where {F,N0,N1,D<:Tuple{Vararg{Integer,N1}}}
  _paramdim(c, Base.front(inputdim), last(inputdim))
end
function paramdim(c::Conv, inputdim::Tuple{Vararg{Integer}})
  _paramdim(c, Base.front(inputdim), last(inputdim))
end

parameter_free(::Conv) = false
function numparam(c::Conv, inputdim::Tuple{Vararg{Integer}})
  nK = ArrayInterface.reduce_tup(*, paramdim(c, inputdim))
  nb = c.outputdim
  nK + nb, getoutputdim(c, inputdim)
end

function getparams(
  c::Conv,
  p::Ptr{T},
  inputdim::Tuple{Vararg{Integer}}
) where {T}
  K = PtrArray(p, paramdim(c, inputdim))
  p += sizeof(T) * length(K)
  b = PtrArray(p, (c.outputdim,))
  (K, b), p + sizeof(T) * length(b)
end

function forward_layer_output_size(::Val{T}, c::Conv, inputdim::Tuple) where {T}
  _, outputdim = numparam(c, inputdim)
  align(static_sizeof(T) * prod(outputdim)), outputdim
end

function init_params!(c::Conv, p, inputdim, rng::AbstractRNG)
  (K, b), p2 = getparams(c, p, inputdim)
  glorot_uniform!(K, rng)
  @turbo warn_check_args = false for i in eachindex(b)
    b[i] = 0
  end
  return p2, getoutputdim(c, inputdim)
end

function alloc_return(outputdim, p)
  R = PtrArray(p, outputdim)
  R, p + align(sizeof(eltype(R)) * length(R))
end

#TODO: DRY with dense
function get∂C(::F, outputdim, ∂Cp::Ptr{T}) where {F,T}
  ∂C = PtrArray(reinterpret(Ptr{T}, ∂Cp), outputdim)
  ∂Cp += align(length(∂C) * sizeof(T))
  ∂C, ∂Cp
end
function get∂C(::F, outputdim, ∂Cp::Ptr{T}, ::False) where {F,T}
  lenC = ArrayInterface.reduce_tup(*, outputdim)
  ∂C = PtrArray(reinterpret(Ptr{T}, ∂Cp), (lenC,))
  ∂Cp += align(lenC * sizeof(T))
  ∂C, ∂Cp
end
function get∂C(::typeof(relu), outputdim, ∂Cp::Ptr)
  ∂C = PtrArray(Ptr{Bit}(∂Cp), outputdim)
  ∂Cp += align((last(static_strides(∂C)) * last(outputdim)) >>> 3)
  ∂C, ∂Cp
end
get∂C(::typeof(identity), _, ∂Cp::Ptr) = nothing, ∂Cp

function (c::Conv)(
  A::AbstractArray{T0},
  p::Ptr{T1},
  pu::Ptr{UInt8}
) where {T0,T1}
  T = promote_type(T0, T1)
  sz = static_size(A)
  outputdim = getoutputdim(c, sz)
  C, pu2 = alloc_return(outputdim, Ptr{T}(pu))
  (K, b), p = getparams(c, p, sz)
  convlayer!(fused_fun(c), C, A, K, b)
  call!(C, unfused_fun(c), p, Ptr{UInt8}(pu2))
end

function valgrad_layer!(
  pg::Ptr{T},
  c::Conv{typeof(identity)},
  A,
  inds,
  p::Ptr{T},
  pu::Ptr{UInt8}
) where {T}
  sz = (Base.front(static_size(A))..., length(inds))
  outputdim = getoutputdim(c, sz)
  R, pu3 = alloc_return(outputdim, Ptr{T}(pu))
  (K, b), p2 = getparams(c, p, sz)
  convlayer!(identity, R, A, K, b, inds)
  pg + (length(K) + length(b)) * sizeof(T), R, p2, Ptr{UInt8}(pu3)
end
function valgrad_layer!(
  pg::Ptr{T},
  c::Conv,
  A,
  inds,
  p::Ptr{T},
  pu::Ptr{UInt8}
) where {T}
  sz = (Base.front(static_size(A))..., length(inds))
  outputdim = getoutputdim(c, sz)
  # we want to allocate ∂C in front of C
  ∂C, pu2 = get∂C(c.f, outputdim, Ptr{T}(pu))
  C, pu3 = alloc_return(outputdim, Ptr{T}(pu2))
  (K, b), p2 = getparams(c, p, sz)
  convlayer!(∂(fused_fun(c)), ∂C, C, A, K, b, inds)
  _valgrad_layer!(
    ∂C,
    C,
    pg + (length(K) + length(b)) * sizeof(T),
    unfused_fun(c),
    C,
    p2,
    Ptr{UInt8}(pu3)
  )
end
function chain_valgrad_entry!(
  pg,
  arg,
  layers::Tuple{Conv,X,Vararg},
  inds,
  p::Ptr,
  pu::Ptr{UInt8}
) where {X}
  l = getfield(layers, 1)
  pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, inds, p, pu)
  val, grad, _ = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
  pullback_param!(pg, l, grad, arg, p, pu)
  return val
end

function valgrad_layer!(
  pg::Ptr{T},
  c::Conv{typeof(identity)},
  A,
  p::Ptr{T},
  pu::Ptr{UInt8}
) where {T}
  sz = static_size(A)
  outputdim = getoutputdim(c, sz)
  R, pu3 = alloc_return(outputdim, Ptr{T}(pu))
  (K, b), p2 = getparams(c, p, sz)
  convlayer!(identity, R, A, K, b)
  pg + (length(K) + length(b)) * sizeof(T), R, p2, Ptr{UInt8}(pu3)
end
function valgrad_layer!(
  pg::Ptr{T},
  c::Conv,
  A,
  p::Ptr{T},
  pu::Ptr{UInt8}
) where {T}
  sz = static_size(A)
  outputdim = getoutputdim(c, sz)
  # we want to allocate ∂C in front of C
  ∂C, pu2 = get∂C(c.f, outputdim, Ptr{T}(pu))
  C, pu3 = alloc_return(outputdim, Ptr{T}(pu2))
  (K, b), p2 = getparams(c, p, sz)
  convlayer!(∂(fused_fun(c)), ∂C, C, A, K, b)
  _valgrad_layer!(
    ∂C,
    C,
    pg + (length(K) + length(b)) * sizeof(T),
    unfused_fun(c),
    C,
    p2,
    Ptr{UInt8}(pu3)
  )
end
function pullback_arg!(
  c::Conv,
  C̄,
  A,
  p::Ptr{T},
  ::Ptr{UInt8},
  pu2::Ptr{UInt8}
) where {T}
  convlayeradjA!(A, first(first(getparams(c, p, static_size(A)))), C̄) # overwrite A
  return A, pu2
end
function pullback_param!(
  pg::Ptr{T},
  c::Conv,
  C̄,
  A,
  ::Ptr{T},
  pu::Ptr{UInt8}
) where {T}
  ∂C = first(get∂C(c.f, static_size(C̄), Ptr{T}(pu)))
  update_C̄!(c.f, C̄, ∂C)
  (gK, gb), _ = getparams(c, pg, static_size(A))
  convlayeradjK!(gK, gb, A, C̄)
  return
end
