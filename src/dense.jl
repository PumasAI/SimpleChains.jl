
"""
    TurboDense{B=true}(activation, outputdim::Integer)

Linear (dense) layer.
- `B` specifies whether the layer includes a bias term.
- The `activation` function is applied elementwise to the result.
- `outputdim` indicates how many dimensions the input is mapped to.

Randomly initializing weights using the (Xaviar) Glorot normal distribution.
"""
struct TurboDense{B,I<:Integer,F}
  f::F
  outputdim::I
end

TurboDense{B}(f::F, t::I) where {F,I<:Integer,B} = TurboDense{B,I,F}(f, static(t))
TurboDense{B}(t::I, f::F) where {F,I<:Integer,B} = TurboDense{B,I,F}(f, static(t))
TurboDense{B,I}(f::F, t::Integer) where {F,I<:Integer,B} = TurboDense{B,I,F}(f, I(t))
function TurboDense{B}(::Integer, ::Integer) where {B}
  throw(
    ArgumentError(
      "TurboDense{$B} requires one integer (output dim) and one function argument.",
    ),
  )
end
TurboDense(f, d) = TurboDense{true}(f, static(d))

function Base.show(io::IO, td::TurboDense{B}) where {B}
  w = B ? "with" : "without"
  print(io, "TurboDense $(td.outputdim) $(w) bias.")
  if td.f !== identity
    println(io)
    show(io, Activation(td.f))
  end
end

# output_dims(d::TurboDense) = getfield(d,:outputdim)

"""
  numparam(d::Layer, inputdim::Tuple)

Returns a `Tuple{Int,S}`.
The first element is the number of parameters required by the layer given an argument of size `inputdim`.
The second argument is the size of the object returned by the layer, which can be fed into `numparam` of
the following layer.
"""
function numparam(d::TurboDense, inputdim::Tuple)
  np = _numparam(d, first(inputdim))
  np, (d.outputdim, Base.tail(inputdim)...)
end
_numparam(d::TurboDense{false}, inputdim::Integer) = inputdim * d.outputdim
_numparam(d::TurboDense{true}, inputdim::Integer) = (inputdim + 1) * d.outputdim
parameter_free(::TurboDense) = false
function layer_output_size(::Val{T}, td::TurboDense, inputdim::Tuple) where {T}
  g1, outputdim = numparam(td, inputdim)
  g2 = prod(outputdim)
  align(static_sizeof(T) * g1) + 2align(static_sizeof(T) * g2), outputdim
end
fast_fuse(td::TurboDense) = fast_fuse(getfield(td, :f))

function getparams(td::TurboDense{false}, p::Ptr{T}, inputdim::Integer) where {T}
  outputdim = td.outputdim
  PtrArray(reinterpret(Ptr{T}, p), (outputdim, inputdim)),
  p + inputdim * outputdim * sizeof(T)
end
function getparams(td::TurboDense{true}, p::Ptr{T}, inputdim::Integer) where {T}
  outputdim = td.outputdim
  inputdimp1 = inputdim + StaticInt(1)
  W = PtrArray(reinterpret(Ptr{T}, p), (outputdim, inputdimp1))
  W, p + (outputdim * inputdimp1) * sizeof(T)
end
function init_params!(td::TurboDense, p, inputdim::Tuple)
  p, outputdim = _init_params!(td, p, first(inputdim))
  p, (outputdim, Base.tail(inputdim)...)
end
function _init_params!(td::TurboDense{true}, p, inputdim::Integer)
  W, p = getparams(td, p, inputdim)
  outputdim = td.outputdim
  glorot_normal!(view(W, :, 1:inputdim))
  @turbo for i = 1:outputdim
    W[i, inputdim+1] = 0
  end
  return p, outputdim
end
function _init_params!(td::TurboDense{false}, p, inputdim::Integer)
  W, p = getparams(td, p, inputdim)
  glorot_normal!(W)
  return p, td.outputdim
end

function alloc_return(
  td::TurboDense,
  batch_size,
  p::Ptr{T},
  ::StaticInt{1},
  ::Tuple{StaticInt{1}},
) where {T}
  O = td.outputdim
  PtrArray(p, (O,)), p + align(O * batch_size * sizeof(T))
end
function alloc_return(
  td::TurboDense,
  batch_size,
  p::Ptr{T},
  ::StaticInt{1},
  ::Tuple{StaticInt{1},StaticInt{2}},
) where {T}
  O = td.outputdim
  PtrArray(p, (O, batch_size)), p + align(O * batch_size * sizeof(T))
end
function alloc_return(
  td::TurboDense,
  batch_size,
  p::Ptr{T},
  ::StaticInt{2},
  ::Tuple{StaticInt{2},StaticInt{1}},
) where {T}
  O = td.outputdim
  PtrArray(p, (batch_size, O))', p + align(O * batch_size * sizeof(T))
end


function (td::TurboDense{O})(
  B::AbstractVecOrMat{T1},
  p::Ptr{T2},
  pu::Ptr{UInt8},
) where {T1,T2,O}
  pB = PtrArray(B)
  T = promote_type(T1, T2)
  GC.@preserve B begin
    put = Base.unsafe_convert(Ptr{T}, pu)
    C, _pu =
      alloc_return(td, size(pB, StaticInt(2)), put, contiguous_axis(B), stride_rank(B))
    pu = Base.unsafe_convert(Ptr{UInt8}, _pu)
    A, p = getparams(td, p, size(B, StaticInt(1)))
    f = td.f
    dense!(f, C, A, pB, static(O), fast_fuse(f))
  end
  C, p, pu
end


function dense!(
  f::F,
  C::AbstractVecOrMat{<:Base.HWReal},
  A::AbstractMatrix{<:Base.HWReal},
  B::AbstractVecOrMat{<:Base.HWReal},
  ::True,
  ::True,
) where {F}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = f(Cmn + A[m, Kp1])
  end
end
function dense!(
  f::F,
  C::AbstractVecOrMat{<:Base.HWReal},
  A::AbstractMatrix{<:Base.HWReal},
  B::AbstractVecOrMat{<:Base.HWReal},
  ::True,
  ::False,
) where {F}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = Cmn + A[m, Kp1]
  end
  @turbo for i ∈ eachindex(C)
    C[i] = f(C[i])
  end
end

function dense!(
  f::F,
  C::AbstractVecOrMat{<:Base.HWReal},
  A::AbstractMatrix{<:Base.HWReal},
  B::AbstractVecOrMat{<:Base.HWReal},
  ::False,
  ::True,
) where {F}
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = f(Cmn)
  end
end
function dense!(
  f::F,
  C::AbstractVecOrMat{<:Base.HWReal},
  A::AbstractMatrix{<:Base.HWReal},
  B::AbstractVecOrMat{<:Base.HWReal},
  ::False,
  ::False,
) where {F}
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = Cmn
  end
  @turbo for i ∈ eachindex(C)
    C[i] = f(C[i])
  end
end

struct ForwardDiffElementwise{F}
  f::F
end
@inline function (fw::ForwardDiffElementwise{F})(x) where {F}
  dx = fw.f(ForwardDiff.Dual(x, one(x)))
  fx = ForwardDiff.value(dx)
  ∂fx = getfield(ForwardDiff.partials(dx).values, 1)
  fx, ∂fx
end
@inline function (fw::ForwardDiffElementwise{typeof(relu)})(x)
  y = zero(x)
  m = x < y
  ifelse(m, y, x), ~m
end
# overloadable
@inline ∂(f::F) where {F} = ForwardDiffElementwise{F}(f)

function get∂C(td::TurboDense{B,D}, C::AbstractArray, ∂Cp::Ptr{UInt8}) where {B,D}
  get∂C(td, C, ∂Cp, fast_fuse(td))
end
function get∂C(::TurboDense, C::AbstractArray{T}, ∂Cp::Ptr{UInt8}, ::True) where {T}
  ∂C = PtrArray(reinterpret(Ptr{T}, ∂Cp), size(C))
  ∂Cp += align(length(∂C) * sizeof(T))
  ∂C, ∂Cp
end
function get∂C(::TurboDense, C::AbstractArray{T}, ∂Cp::Ptr{UInt8}, ::False) where {T}
  lenC = length(C)
  ∂C = PtrArray(reinterpret(Ptr{T}, ∂Cp), (lenC,))
  ∂Cp += align(lenC * sizeof(T))
  ∂C, ∂Cp
end
function get∂C(
  ::TurboDense{B,D,typeof(relu)},
  C::AbstractArray,
  ∂Cp::Ptr{UInt8},
) where {B,D}
  outputdim = size(C)
  ∂C = PtrArray(reinterpret(Ptr{Bit}, ∂Cp), outputdim)
  ∂Cp += align((last(StrideArraysCore.strides(∂C)) >>> 3) * last(outputdim))
  ∂C, ∂Cp
end
function get∂C(
  ::TurboDense{B,D,typeof(identity)},
  ::AbstractArray,
  ∂Cp::Ptr{UInt8},
) where {B,D}
  (nothing, ∂Cp)
end

function dense!(
  f::F,
  ∂C::AbstractArray{T1,N},
  C::AbstractArray{T2,N},
  A::AbstractMatrix,
  B::AbstractArray{T3,N},
  ::True,
) where {F,T1<:Base.HWReal,T2<:Base.HWReal,T3<:Base.HWReal,N}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  ∂f = ∂(f)
  @turbo for n ∈ indices((B, C, ∂C), 2), m ∈ indices((A, C, ∂C), 1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m, k] * B[k, n]
    end
    Cmn, ∂Cmn = ∂f(Cmn + A[m, Kp1])
    ∂C[m, n] = ∂Cmn
    C[m, n] = Cmn
  end
end
function dense!(
  f::F,
  ∂C::AbstractVector{<:Base.HWReal},
  C::AbstractMatrix{<:Base.HWReal},
  A::AbstractMatrix{<:Base.HWReal},
  B::AbstractMatrix{<:Base.HWReal},
  ::True,
) where {F}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = Cmn + A[m, Kp1]
  end
  ∂f = ∂(f)
  @turbo for i ∈ eachindex(C)
    Cᵢ, ∂Cᵢ = ∂f(C[i])
    ∂C[i] = ∂Cᵢ
    C[i] = Cᵢ
  end
end

function dense!(
  f::F,
  ∂C::AbstractArray{T1,N},
  C::AbstractArray{T2,N},
  A::AbstractMatrix,
  B::AbstractArray{T3,N},
  ::False,
) where {F,T1<:Base.HWReal,T2<:Base.HWReal,T3<:Base.HWReal,N}
  ∂f = ∂(f)
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, n]
    end
    Cmn, ∂Cmn = ∂f(Cmn)
    C[m, n] = Cmn
    ∂C[m, n] = ∂Cmn
  end
end
function dense!(
  f::F,
  ∂C::AbstractVector{<:Base.HWReal},
  C::AbstractMatrix{<:Base.HWReal},
  A::AbstractMatrix{<:Base.HWReal},
  B::AbstractMatrix{<:Base.HWReal},
  ::False,
) where {F}
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = Cmn
  end
  ∂f = ∂(f)
  @turbo for i ∈ eachindex(C)
    Cᵢ, ∂Cᵢ = ∂f(C[i])
    C[i] = Cᵢ
    ∂C[i] = ∂Cᵢ
  end
end

function dense!(
  ::Union{typeof(tanh_fast),typeof(tanh)},
  ∂C::AbstractVector{<:Base.HWReal},
  C::AbstractMatrix{<:Base.HWReal},
  A::AbstractMatrix{<:Base.HWReal},
  B::AbstractMatrix{<:Base.HWReal},
  ::True,
)
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = Cmn + A[m, Kp1]
  end
  @turbo for i ∈ eachindex(C)
    Cᵢ = tanh_fast(C[i])
    C[i] = Cᵢ
    ∂C[i] = one(Cᵢ) - Cᵢ * Cᵢ
  end
end
function dense!(
  ::Union{typeof(tanh_fast),typeof(tanh)},
  ∂C::AbstractVector{<:Base.HWReal},
  C::AbstractMatrix{<:Base.HWReal},
  A::AbstractMatrix{<:Base.HWReal},
  B::AbstractMatrix{<:Base.HWReal},
  ::False,
)
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = Cmn
  end
  @turbo for i ∈ eachindex(C)
    Cᵢ = tanh_fast(C[i])
    C[i] = Cᵢ
    ∂C[i] = one(Cᵢ) - Cᵢ * Cᵢ
  end
end
function dense!(
  ::typeof(relu),
  ∂C::AbstractArray{Bool,N},
  C::AbstractArray{T1,N},
  A::AbstractMatrix,
  B::AbstractArray{T2,N},
  ::True,
) where {T1<:Base.HWReal,T2<:Base.HWReal,N}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m, k] * B[k, n]
    end
    Cmnr = Cmn + A[m, Kp1]
    Cmnr_gt_0 = Cmnr > zero(Cmnr)
    C[m, n] = ifelse(Cmnr_gt_0, Cmnr, zero(Cmnr))
    ∂C[m, n] = Cmnr_gt_0
  end
end
function dense!(
  ::typeof(relu),
  ∂C::AbstractArray{Bool,N},
  C::AbstractArray{T1,N},
  A::AbstractMatrix,
  B::AbstractArray{T2,N},
  ::False,
) where {T1<:Base.HWReal,T2<:Base.HWReal,N}
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, n]
    end
    Cmn_gt_0 = Cmn > zero(Cmn)
    C[m, n] = ifelse(Cmn_gt_0, Cmn, zero(Cmn))
    ∂C[m, n] = Cmn_gt_0
  end
end
function dense!(
  ::typeof(identity),
  ::Nothing,
  C::AbstractArray{T1,N},
  A::AbstractMatrix,
  B::AbstractArray{T2,N},
  ::True,
) where {T1<:Base.HWReal,T2<:Base.HWReal,N}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = Cmn + A[m, Kp1]
  end
end
function dense!(
  ::typeof(identity),
  ::Nothing,
  C::AbstractArray{T1,N},
  A::AbstractMatrix,
  B::AbstractArray{T2,N},
  ::False,
) where {T1<:Base.HWReal,T2<:Base.HWReal,N}
  @turbo for n ∈ indices((B, C), 2), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, n]
    end
    C[m, n] = Cmn
  end
end

#=
function dense!(
  f::F,
  _∂C::AbstractArray{T1,N},
  _C::AbstractArray{T2,N},
  _A::AbstractMatrix,
  _B::AbstractArray{T3,N},
  inds::AbstractVector{<:Integer},
  ::True,
) where {F,T1<:Base.HWReal,T2<:Base.HWReal,T3<:Base.HWReal,N}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  K = size(A, StaticInt(2)) - StaticInt(1)
  ∂f = ∂(f)
  @turbo for n ∈ indices((inds, C, ∂C), (1,2,2)), m ∈ indices((A, C, ∂C), 1)
    Cmn = zero(eltype(C))
    for k ∈ CloseOpen(K)
      Cmn += A[m, k] * B[k, inds[n]]
    end
    Cmn, ∂Cmn = ∂f(Cmn + A[m, K])
    ∂C[m, n] = ∂Cmn
    C[m, n] = Cmn
  end
end
function dense!(
  f::F,
  _∂C::AbstractVector{<:Base.HWReal},
  _C::AbstractMatrix{<:Base.HWReal},
  _A::AbstractMatrix{<:Base.HWReal},
  _B::AbstractMatrix{<:Base.HWReal},
  inds::AbstractVector{<:Integer},
  ::True,
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  K = size(A, StaticInt(2)) - StaticInt(1)
  @turbo for n ∈ indices((inds, C), (1,2)), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ CloseOpen(K)
      Cmn += A[m, k] * B[k, inds[n]]
    end
    C[m, n] = Cmn + A[m, K]
  end
  ∂f = ∂(f)
  @turbo for i ∈ eachindex(C)
    Cᵢ, ∂Cᵢ = ∂f(C[i])
    ∂C[i] = ∂Cᵢ
    C[i] = Cᵢ
  end
end

function dense!(
  f::F,
  _∂C::AbstractArray{T1,N},
  _C::AbstractArray{T2,N},
  _A::AbstractMatrix,
  _B::AbstractArray{T3,N},
  inds::AbstractVector{<:Integer},
  ::False,
) where {F,T1<:Base.HWReal,T2<:Base.HWReal,T3<:Base.HWReal,N}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  ∂f = ∂(f)
  @turbo for n ∈ indices((inds, C), (1,2)), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, inds[n]]
    end
    Cmn, ∂Cmn = ∂f(Cmn)
    C[m, n] = Cmn
    ∂C[m, n] = ∂Cmn
  end
end
function dense!(
  f::F,
  _∂C::AbstractVector{<:Base.HWReal},
  _C::AbstractMatrix{<:Base.HWReal},
  _A::AbstractMatrix{<:Base.HWReal},
  _B::AbstractMatrix{<:Base.HWReal},
  inds::AbstractVector{<:Integer},
  ::False,
) where {F}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  @turbo for n ∈ indices((inds, C), (1,2)), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, inds[n]]
    end
    C[m, n] = Cmn
  end
  ∂f = ∂(f)
  @turbo for i ∈ eachindex(C)
    Cᵢ, ∂Cᵢ = ∂f(C[i])
    C[i] = Cᵢ
    ∂C[i] = ∂Cᵢ
  end
end

function dense!(
  ::Union{typeof(tanh_fast),typeof(tanh)},
  _∂C::AbstractVector{<:Base.HWReal},
  _C::AbstractMatrix{<:Base.HWReal},
  _A::AbstractMatrix{<:Base.HWReal},
  _B::AbstractMatrix{<:Base.HWReal},
  inds::AbstractVector{<:Integer},
  ::True,
)
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  K = size(A, StaticInt(2)) - StaticInt(1)
  @turbo for n ∈ indices((inds, C), (1,2)), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ CloseOpen(K)
      Cmn += A[m, k] * B[k, inds[n]]
    end
    C[m, n] = Cmn + A[m, K]
  end
  Cv = zero_offsets(vec(C))
  @turbo for i ∈ eachindex(Cv)
    Cᵢ = tanh_fast(Cv[i])
    Cv[i] = Cᵢ
    ∂C[i] = one(Cᵢ) - Cᵢ * Cᵢ
  end
end
function dense!(
  ::Union{typeof(tanh_fast),typeof(tanh)},
  _∂C::AbstractVector{<:Base.HWReal},
  _C::AbstractMatrix{<:Base.HWReal},
  _A::AbstractMatrix{<:Base.HWReal},
  _B::AbstractMatrix{<:Base.HWReal},
  inds::AbstractVector{<:Integer},
  ::False,
)
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  @turbo for n ∈ indices((inds, C), (1,2)), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, inds[n]]
    end
    C[m, n] = Cmn
  end
  @turbo for i ∈ eachindex(C)
    Cᵢ = tanh_fast(C[i])
    C[i] = Cᵢ
    ∂C[i] = one(Cᵢ) - Cᵢ * Cᵢ
  end
end
function dense!(
  ::typeof(relu),
  _∂C::AbstractArray{Bool,N},
  _C::AbstractArray{T1,N},
  _A::AbstractMatrix,
  _B::AbstractArray{T2,N},
  inds::AbstractVector{<:Integer},
  ::True,
) where {T1<:Base.HWReal,T2<:Base.HWReal,N}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  K = size(A, StaticInt(2)) - StaticInt(1)
  @turbo for n ∈ indices((inds, C), (1,2)), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ CloseOpen(K)
      Cmn += A[m, k] * B[k, inds[n]]
    end
    Cmnr = Cmn + A[m, K]
    Cmnr_gt_0 = Cmnr > zero(Cmnr)
    C[m, n] = ifelse(Cmnr_gt_0, Cmnr, zero(Cmnr))
    ∂C[m, n] = Cmnr_gt_0
  end
end
function dense!(
  ::typeof(relu),
  _∂C::AbstractArray{Bool,N},
  _C::AbstractArray{T1,N},
  _A::AbstractMatrix,
  _B::AbstractArray{T2,N},
  inds::AbstractVector{<:Integer},
  ::False,
) where {T1<:Base.HWReal,T2<:Base.HWReal,N}
  ∂C = zero_offsets(_∂C)
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  @turbo for n ∈ indices((inds, C), (1,2)), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, inds[n]]
    end
    Cmn_gt_0 = Cmn > zero(Cmn)
    C[m, n] = ifelse(Cmn_gt_0, Cmn, zero(Cmn))
    ∂C[m, n] = Cmn_gt_0
  end
end
function dense!(
  ::typeof(identity),
  ::Nothing,
  _C::AbstractArray{T1,N},
  _A::AbstractMatrix,
  _B::AbstractArray{T2,N},
  inds::AbstractVector{<:Integer},
  ::True,
) where {T1<:Base.HWReal,T2<:Base.HWReal,N}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  K = ArrayInterface.size(A, StaticInt(2)) - One()
  @turbo for n ∈ indices((inds, C), (1,2)), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ CloseOpen(K)
      Cmn += A[m, k] * B[k, inds[n]]
    end
    C[m, n] = Cmn + A[m, K]
  end
end
function dense!(
  ::typeof(identity),
  ::Nothing,
  _C::AbstractArray{T1,N},
  _A::AbstractMatrix,
  _B::AbstractArray{T2,N},
  inds::AbstractVector{<:Integer},
  ::False,
) where {T1<:Base.HWReal,T2<:Base.HWReal,N}
  C = zero_offsets(_C)
  A = zero_offsets(_A)
  B = zero_offsets(_B)
  @turbo for n ∈ indices((inds, C), (1,2)), m ∈ indices((A, C), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A, B), (2, 1))
      Cmn += A[m, k] * B[k, inds[n]]
    end
    C[m, n] = Cmn
  end
end

function valgrad_layer!(
  pg::Ptr{T},
  td::TurboDense{O},
  B, inds,
  p::Ptr{T},
  pu::Ptr{UInt8},
) where {T,O}
  input_dim = size(B, StaticInt(1))
  batch_size = size(B, StaticInt(2))
  pu2 = Base.unsafe_convert(Ptr{T}, pu + align(batch_size * td.outputdim * sizeof(T)))
  C, _pu3 = alloc_return(td, batch_size, pu2, contiguous_axis(B), stride_rank(B))
  pu3 = Base.unsafe_convert(Ptr{UInt8}, _pu3)
  ∂C, _ = get∂C(td, C, pu)
  A, p2 = getparams(td, p, input_dim)
  f = td.f
  dense!(f, ∂C, C, A, B, inds, static(O))
  # doesn'tneed a pullback
  pg + length(A) * sizeof(T), C, p2, pu3
end
function chain_valgrad_entry!(
  pg,
  arg,
  layers::Tuple{TurboDense,X,Vararg},
  inds,
  p::Ptr,
  pu::Ptr{UInt8},
) where {X}
  l = getfield(layers, 1)
  pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, inds, p, pu)
  val, grad, _ = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
  pullback_param!(pg, l, grad, arg, p, pu)
  return val
end
=#


function valgrad_layer!(
  pg::Ptr{T},
  td::TurboDense{O},
  B,
  p::Ptr{T},
  pu::Ptr{UInt8},
) where {T,O}
  input_dim = size(B, StaticInt(1))
  batch_size = size(B, StaticInt(2))
  pu2 = Base.unsafe_convert(Ptr{T}, pu + align(batch_size * td.outputdim * sizeof(T)))
  C, _pu3 = alloc_return(td, batch_size, pu2, contiguous_axis(B), stride_rank(B))
  pu3 = Base.unsafe_convert(Ptr{UInt8}, _pu3)
  ∂C, _ = get∂C(td, C, pu)
  A, p2 = getparams(td, p, input_dim)
  f = td.f
  dense!(f, ∂C, C, A, B, static(O))
  # doesn'tneed a pullback
  pg + length(A) * sizeof(T), C, p2, pu3
end
alloc_return_B_dense(B::PtrArray, pu::Ptr{UInt8}, _) = (B, pu) # assume `PtrArray` means we can destroy it
function alloc_return_B_dense(B::AbstractArray{T}, pu::Ptr{UInt8}, input_dim) where {T}
  si = bytestrideindex(B)
  sp = stridedpointer(reinterpret(Ptr{T}, pu), si)
  B̄ = PtrArray(sp, (input_dim, size(B, static(2))), StrideArraysCore.val_dense_dims(B))
  B̄, pu + align(length(B̄) * sizeof(T))
end
function pullback!(
  pg::Ptr{T},
  td::TurboDense{O},
  C̄,
  B,
  p::Ptr{T},
  pu::Ptr{UInt8},
  pu2::Ptr{UInt8},
) where {T,O}
  # Start with 4-arg `pulback!` to update `∂C`
  pullback_param!(pg, td, C̄, B, p, pu) # Ā = C̄ * B'
  # Now 5-arg
  # B̄ = A' * C̄
  intput_dims = size(B, StaticInt(1))
  A, _ = getparams(td, p, intput_dims)
  B̄, pu2 = alloc_return_B_dense(B, pu2, intput_dims)
  dense!(identity, nothing, B̄, matrix_view(td, A)', C̄, False())
  B̄, pu2
end
matrix_view(::TurboDense{false}, A) = A
function matrix_view(::TurboDense{true}, A)
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  view(A, :, static(1):K)
end
update_C̄!(::typeof(identity), _, __) = nothing #=∂C=#
function update_C̄!(::F, C̄, ∂C) where {F}
  @turbo for i ∈ eachindex(∂C)
    C̄[i] *= ∂C[i]
  end
end
function pullback_param!(
  pg::Ptr{T},
  td::TurboDense{O},
  C̄,
  B,
  ::Ptr{T},
  pu::Ptr{UInt8},
) where {T,O}
  # Ā = C̄ * B'
  ∂C = first(get∂C(td, C̄, pu))
  update_C̄!(td.f, C̄, ∂C)
  Ā, __ = getparams(td, pg, size(B, StaticInt(1)))
  dense_param_update!(td, Ā, C̄, B)
  return nothing
end
function dense_param_update!(::TurboDense{true}, Ā, C̄, B)
  Kp1 = ArrayInterface.size(Ā, StaticInt(2))
  K = Kp1 - StaticInt(1)
  dense!(identity, nothing, view(Ā, :, static(1):K), C̄, B', False())
  @turbo for m ∈ axes(Ā, 1)
    s = zero(eltype(Ā))
    for n ∈ axes(C̄, 2)
      s += C̄[m, n]
    end
    Ā[m, Kp1] = s
  end
end
function dense_param_update!(::TurboDense{false}, Ā, C̄, B)
  dense!(identity, nothing, Ā, C̄, B', False())
end
struct DualCall{F}
  f::F
end
@generated function (dc::DualCall)(x::T, y::Vararg{T,P}) where {P,T}
  quote
    $(Expr(:meta, :inline))
    @inbounds d = Base.Cartesian.@ncall $P ForwardDiff.Dual x p -> y[p]
    fd = dc.f(d)
    ∂fd = ForwardDiff.partials(fd)
    @inbounds (ForwardDiff.value(fd), (Base.Cartesian.@ntuple $P p -> ∂fd[p])...)
  end
end
dualeval!(
  ::typeof(identity),
  ::AbstractArray{D},
) where {T,P,D<:ForwardDiff.Dual{<:Any,T,P}} = nothing
@generated function dualeval!(
  f::F,
  Cdual::AbstractArray{D},
) where {F,T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  if isa(T, Base.HWReal)
    quote
      C = reinterpret(reshape, T, Cdual)
      g = DualCall(f)
      @turbo warn_check_args = false for m ∈ eachindex(Cdual)
        (Base.Cartesian.@ntuple $(P + 1) p -> C[p, m]) =
          Base.Cartesian.@ncall $(P + 1) g p -> C[p, m]
      end
    end
  else
    quote
      @inbounds for i ∈ eachindex(Cdual)
        Cdual[i] = f(Cdual[i])
      end
    end
  end
end

# maybe collapses dims 1 and 2 of a 3d array.
_collapse_dims12(::AbstractArray, A) = A
function collapse_dims12(A::PtrArray{S,(true, true),T,1,1,0,(1, 2)}) where {S,T}
  M, N = size(A)
  sp = stridedpointer(A)
  o1, o2 = offsets(sp)
  x1, _ = strides(sp)
  si = ArrayInterface.StrideIndex{2,(1, 2),1}((x1, StaticInt(0)), (o1, o2))
  spnew = stridedpointer(pointer(sp), si)
  PtrArray(spnew, (M * N, StaticInt(1)), Val((true, true)))
end
function collapse_dims12(A::PtrArray{S,(true, true, true),T,3,1,0,(1, 2, 3)}) where {S,T}
  M, N, P = size(A)
  sp = stridedpointer(A)
  o1, o2, o3 = offsets(sp)
  x1, _, x3 = strides(sp)
  si = ArrayInterface.StrideIndex{3,(1, 2, 3),1}((x1, StaticInt(0), x3), (o1, o2, o3))
  spnew = stridedpointer(pointer(sp), si)
  PtrArray(spnew, (M * N, StaticInt(1), P), Val((true, true, true)))
end
const Collapsible12PtrArray{S,T} = Union{
  PtrArray{S,(true, true),T,1,1,0,(1, 2)},
  PtrArray{S,(true, true, true),T,3,1,0,(1, 2, 3)},
}
function _collapse_dims12(A::Collapsible12PtrArray, O::AbstractArray)
  StrideArray(collapse_dims12(A), O)
end
function collapse_dims12(A::AbstractArray)
  _collapse_dims12(PtrArray(A), A)
end
collapse_dims12(::AbstractArray, ::AbstractArray, A, B) = A, B
function collapse_dims12(A::Collapsible12PtrArray, B::Collapsible12PtrArray, OA, OB)
  StrideArray(collapse_dims12(A), OA), StrideArray(collapse_dims12(B), OB)
end
function collapse_dims12(A::Collapsible12PtrArray, B::Collapsible12PtrArray)
  collapse_dims12(A), collapse_dims12(B)
end
function collapse_dims12(A::AbstractArray, B::AbstractArray)
  collapse_dims12(PtrArray(A), PtrArray(B), A, B)
end

@inline dual_eltype(
  ::Type{ForwardDiff.Dual{T,V,P}},
) where {T,V<:Union{Bool,Base.HWReal},P} = V
@inline dual_eltype(::Type{ForwardDiff.Dual{T,V,P}}) where {T,V<:ForwardDiff.Dual,P} =
  dual_eltype(V)
@inline reinterpret_dual(A::AbstractArray{D}) where {D<:ForwardDiff.Dual} =
  reinterpret(reshape, dual_eltype(D), A)

function matmul!(
  Cdual::AbstractVector{D},
  Adual::AbstractMatrix{D},
  B::AbstractVector,
  ::True,
) where {D<:ForwardDiff.Dual}
  C = reinterpret_dual(Cdual)
  A = reinterpret_dual(Adual)
  Kp1 = ArrayInterface.size(A, StaticInt(3))
  K = Kp1 - StaticInt(1)
  Af, Cf = collapse_dims12(A, C)
  @turbo for m ∈ indices((Af, Cf), 2), p ∈ indices((Af, Cf), 1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += Af[p, m, k] * B[k]
    end
    Cf[p, m] = Cmn + Af[p, m, Kp1]
  end
end
function matmul!(
  Cdual::AbstractMatrix{D},
  Adual::AbstractMatrix{D},
  B::AbstractMatrix,
  ::True,
) where {D<:ForwardDiff.Dual}
  C = reinterpret_dual(Cdual)
  A = reinterpret_dual(Adual)
  Kp1 = ArrayInterface.size(A, StaticInt(3))
  K = Kp1 - StaticInt(1)
  Af, Cf = collapse_dims12(A, C)
  @turbo for n ∈ indices((B, Cf), (2, 3)),
    m ∈ indices((Af, Cf), 2),
    p ∈ indices((Af, Cf), 1)

    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += Af[p, m, k] * B[k, n]
    end
    Cf[p, m, n] = Cmn + Af[p, m, Kp1]
  end
end
function matmul!(
  Cdual::AbstractVector{D},
  Adual::AbstractMatrix{D},
  B::AbstractVector,
  ::False,
) where {D<:ForwardDiff.Dual}
  C = reinterpret_dual(Cdual)
  A = reinterpret_dual(Adual)
  Af, Cf = collapse_dims12(A, C)
  @turbo for m ∈ indices((Af, Cf), 2), p ∈ indices((Af, Cf), 1)
    Cmn = zero(eltype(C))
    for k ∈ indices((Af, B), (3, 1))
      Cmn += Af[p, m, k] * B[k]
    end
    Cf[p, m] = Cmn
  end
end
function matmul!(
  Cdual::AbstractMatrix{D},
  Adual::AbstractMatrix{D},
  B::AbstractMatrix,
  ::False,
) where {D<:ForwardDiff.Dual}
  C = reinterpret_dual(Cdual)
  A = reinterpret_dual(Adual)
  Af, Cf = collapse_dims12(A, C)
  @turbo for n ∈ indices((B, Cf), (2, 3)),
    m ∈ indices((Af, Cf), 2),
    p ∈ indices((Af, Cf), 1)

    Cmn = zero(eltype(C))
    for k ∈ indices((Af, B), (3, 1))
      Cmn += Af[p, m, k] * B[k, n]
    end
    Cf[p, m, n] = Cmn
  end
end

# @inline function clip_r1_view(a::AbstractVector)
#   d1 = axes(a, StaticInt(1))
#   view(a, static_first(d1) + static_step(d1):static_last(d1))
# end
# @inline function clip_r1_view(A::AbstractMatrix)
#   d1 = axes(A, StaticInt(1))
#   view(A, static_first(d1) + static_step(d1):static_last(d1), :)
# end


# if `A` and `B` both contain dual numbers,
# we currently peal through just a single layer of duals
# If the duals are not nested, this should allow `@turbo`
# to work. Otherwise, the fallback loop should run.
# TODO: supported `@turbo`-ing nested duals here.
function matmul!(
  Cdual::AbstractVector{D},
  _A::AbstractMatrix,
  Bdual::AbstractVector{D},
  ::True,
) where {T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = zero_offsets(reinterpret(reshape, T, Cdual))
  A = zero_offsets(_A)
  B = zero_offsets(reinterpret(reshape, T, Bdual))
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo warn_check_args = false for p ∈ indices((B, C), 1), m ∈ indices((A, C), (1, 2))

    Cmn = zero(T)
    for k ∈ CloseOpen(K)
      Cmn += A[m, k] * B[p, k]
    end
    C[p, m] = Cmn + (A[m, K] * (p == 0))
  end
end
function matmul!(
  Cdual::AbstractMatrix{D},
  _A::AbstractMatrix,
  Bdual::AbstractMatrix{D},
  ::True,
) where {T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = zero_offsets(reinterpret(reshape, T, Cdual))
  A = zero_offsets(_A)
  B = zero_offsets(reinterpret(reshape, T, Bdual))
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo warn_check_args = false for p ∈ indices((B, C), 1),
    n ∈ indices((B, C), 3),
    m ∈ indices((A, C), (1, 2))

    Cmn = zero(T)
    for k ∈ CloseOpen(K)
      Cmn += A[m, k] * B[p, k, n]
    end
    C[p, m, n] = Cmn + (A[m, K] * (p == 0))
  end
end
function matmul!(
  Cdual::AbstractVector{D},
  A::AbstractMatrix,
  Bdual::AbstractVector{D},
  ::False,
) where {T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  B = reinterpret(reshape, T, Bdual)
  @turbo warn_check_args = false for m ∈ indices((A, C), (1, 2)), p ∈ indices((B, C), 1)
    Cpmn = zero(T)
    for k ∈ indices((A, B), 2)
      Cpmn += A[m, k] * B[p, k]
    end
    C[p, m] += Cpmn
  end
end
function matmul!(
  Cdual::AbstractMatrix{D},
  A::AbstractMatrix,
  Bdual::AbstractMatrix{D},
  ::False,
) where {T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  B = reinterpret(reshape, T, Bdual)
  @turbo warn_check_args = false for n ∈ indices((B, C), 3),
    m ∈ indices((A, C), (1, 2)),
    p ∈ indices((B, C), 1)

    Cpmn = zero(T)
    for k ∈ indices((A, B), 2)
      Cpmn += A[m, k] * B[p, k, n]
    end
    C[p, m, n] += Cpmn
  end
end

function matmul!(
  Cdual::AbstractVector{D},
  Adual::AbstractMatrix{D},
  Bdual::AbstractVector{D},
  ::True,
) where {T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  B = reinterpret(reshape, T, Bdual)
  Kp1 = ArrayInterface.size(A, StaticInt(3))
  K = Kp1 - StaticInt(1)
  Pstatic = StaticInt(P)
  Af, Cf1 = collapse_dims12(A, C)
  @turbo warn_check_args = false for m ∈ indices((Af, Cf1), 2), p ∈ indices((Af, Cf1), 1)
    Cmn = zero(T)
    for k ∈ 1:K
      Cmn += Af[p, m, k] * B[1, k]
    end
    Cf1[p, m] = Cmn + Af[p, m, Kp1]
  end
  @turbo warn_check_args = false for m ∈ indices((A, C), 2), p ∈ 1:Pstatic, k ∈ 1:K

    C[p+1, m] += A[1, m, k] * B[p+1, k]
  end
end
function matmul!(
  Cdual::AbstractMatrix{D},
  Adual::AbstractMatrix{D},
  Bdual::AbstractMatrix{D},
  ::True,
) where {T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  B = reinterpret(reshape, T, Bdual)
  Kp1 = ArrayInterface.size(A, StaticInt(3))
  K = Kp1 - StaticInt(1)
  Pstatic = StaticInt(P)
  Af, Cf1 = collapse_dims12(A, C)
  @turbo warn_check_args = false for n ∈ indices((B, Cf1), 3),
    m ∈ indices((Af, Cf1), 2),
    p ∈ indices((Af, Cf1), 1)

    Cmn = zero(T)
    for k ∈ 1:K
      Cmn += Af[p, m, k] * B[1, k, n]
    end
    Cf1[p, m, n] = Cmn + Af[p, m, Kp1]
  end
  # Bf, Cf2 = collapse_dims12(clip_r1_view(B), clip_r1_view(C))
  @turbo warn_check_args = false for n ∈ indices((B, C), 3),
    m ∈ indices((A, C), 2),
    p ∈ 1:Pstatic,
    k ∈ 1:K

    C[p+1, m, n] += A[1, m, k] * B[p+1, k, n]
  end
end
function matmul!(
  Cdual::AbstractVector{D},
  Adual::AbstractMatrix{D},
  Bdual::AbstractVector{D},
  ::False,
) where {T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  B = reinterpret(reshape, T, Bdual)
  Pstatic = StaticInt(P)
  Af, Cf1 = collapse_dims12(A, C)
  @turbo warn_check_args = false for m ∈ indices((Af, Cf1), 2), p ∈ indices((Af, Cf1), 1)
    Cmn = zero(T)
    for k ∈ indices((Af, B), (3, 2))
      Cmn += Af[p, m, k] * B[1, k]
    end
    Cf1[p, m] = Cmn
  end
  @turbo warn_check_args = false for m ∈ indices((A, C), 2),
    p ∈ 1:Pstatic,
    k ∈ indices((A, B), (3, 2))

    C[p+1, m] += A[1, m, k] * B[p+1, k]
  end
end


function matmul!(
  Cdual::AbstractMatrix{D},
  Adual::AbstractMatrix{D},
  Bdual::AbstractMatrix{D},
  ::False,
) where {T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  B = reinterpret(reshape, T, Bdual)
  Pstatic = StaticInt(P)
  Af, Cf1 = collapse_dims12(A, C)
  @turbo warn_check_args = false for n ∈ indices((B, Cf1), 3),
    m ∈ indices((Af, Cf1), 2),
    p ∈ indices((Af, Cf1), 1)

    Cmn = zero(T)
    for k ∈ indices((Af, B), (3, 2))
      Cmn += Af[p, m, k] * B[1, k, n]
    end
    Cf1[p, m, n] = Cmn
  end
  # Bf, Cf2 = collapse_dims12(clip_r1_view(B), clip_r1_view(C))
  @turbo warn_check_args = false for n ∈ indices((B, C), 3),
    m ∈ indices((A, C), 2),
    p ∈ 1:Pstatic,
    k ∈ indices((A, B), (3, 2))

    C[p+1, m, n] += A[1, m, k] * B[p+1, k, n]
  end
end

function dense!(
  f::F,
  Cdual::AbstractArray{D},
  A::AbstractMatrix,
  B::AbstractArray,
  ::BT,
  ::FF,
) where {F,BT,FF,T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  matmul!(Cdual, A, B, BT())
  dualeval!(f, Cdual)
end
