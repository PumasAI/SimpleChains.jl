struct TurboDense{B,D<:Tuple{<:Integer,<:Integer},F}
  f::F
  dims::D
end

TurboDense{B}(f::F, t::Tuple{I1,I2}) where {F,I1,I2,B} = TurboDense{B,Tuple{I1,I2},F}(f, t)
TurboDense(f::F, t::Tuple{I1,I2}) where {F,I1,I2} = TurboDense{true,Tuple{I1,I2},F}(f, t)

input_dims(d::TurboDense) = getfield(d.dims, 1)
function numparam(d::TurboDense{false})
  id,  od = d.dims
  id * od
end
function numparam(d::TurboDense{true})
  id,  od = d.dims
  id * od + od
end
function output_size(::Val{T}, td::TurboDense, s) where {T}
  g1 = numparam(td) # for gradients
  g2 = getfield(td.dims, 1) * getfield(s, 2) # for output
  align(static_sizeof(T) * g1) + align(static_sizeof(T) * g2), (getfield(td.dims, 1), getfield(s,2))
end
fast_fuse(::typeof(relu)) = True()
fast_fuse(::typeof(abs)) = True()
fast_fuse(::typeof(abs2)) = True()
fast_fuse(::typeof(identity)) = True()
fast_fuse(_) = False()
fast_fuse(td::TurboDense) = fast_fuse(getfield(td,:f))

function getparams(td::TurboDense{false}, p::Ptr{T}) where {T}
  id, od = td.dims
  PtrArray(reinterpret(Ptr{T}, p), (od, id)), p + id * od * sizeof(T)
end
function getparams(td::TurboDense{true}, p::Ptr{T}) where {T}
  id, od = td.dims
  idp1 = id + StaticInt(1)
  W = PtrArray(reinterpret(Ptr{T}, p), (od, idp1))
  W, p + (od * idp1) * sizeof(T)
end
function init_params!(td::TurboDense{true}, p)
  W, p = getparams(td, p)
  id, od = td.dims
  lrng = local_rng()
  randn!(lrng, view(W, :, 1:id), static(0), static(0), Base.FastMath.sqrt_fast(eltype(W)(2/(id+od))))
  # randn!(lrng, view(W, :, id+1))
  fill!(view(W, :, id+1), 0)
  return p
end
function init_params!(td::TurboDense{false}, p)
  W, p = getparams(td, p)
  randn!(local_rng(), W, static(0), static(0), Base.FastMath.sqrt_fast(eltype(W)(2/sum(td.dims))))
  return p
end

function alloc_return(td::TurboDense, batch_size, p::Ptr{T}, ::StaticInt{1}, ::Tuple{StaticInt{1}}) where {T}
  O = getfield(td.dims,2)
  PtrArray(p, (O, )), p + align(O*batch_size*sizeof(T))
end
function alloc_return(td::TurboDense, batch_size, p::Ptr{T}, ::StaticInt{1}, ::Tuple{StaticInt{1},StaticInt{2}}) where {T}
  O = getfield(td.dims,2)
  PtrArray(p, (O, batch_size)), p + align(O*batch_size*sizeof(T))
end
function alloc_return(td::TurboDense, batch_size, p::Ptr{T}, ::StaticInt{2}, ::Tuple{StaticInt{2},StaticInt{1}}) where {T}
  O = getfield(td.dims,2)
  PtrArray(p, (batch_size,O))', p + align(O*batch_size*sizeof(T))
end


function (td::TurboDense{O})(B::AbstractVecOrMat{T1}, p::Ptr{T2}, pu::Ptr{UInt8}) where {T1,T2,O}
  pB = PtrArray(B)
  T = promote_type(T1, T2)
  GC.@preserve B begin
    C, _pu = alloc_return(td, size(pB, StaticInt(2)), Base.unsafe_convert(Ptr{T}, pu), contiguous_axis(B), stride_rank(B))
    pu = Base.unsafe_convert(Ptr{UInt8}, _pu)
    A, p = getparams(td, p)
    f = td.f
    dense!(f, C, A, pB, static(O), fast_fuse(f))
  end
  C, p, pu
end


function dense!(f::F, C::AbstractVecOrMat{<:Base.HWReal}, A::AbstractMatrix{<:Base.HWReal}, B::AbstractVecOrMat{<:Base.HWReal}, ::True, ::True) where {F}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = f(Cmn + A[m,Kp1])
  end
end
function dense!(f::F, C::AbstractVecOrMat{<:Base.HWReal}, A::AbstractMatrix{<:Base.HWReal}, B::AbstractVecOrMat{<:Base.HWReal}, ::True, ::False) where {F}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn + A[m,Kp1]
  end
  @turbo for i ∈ eachindex(C)
    C[i] = f(C[i])
  end
end

function dense!(f::F, C::AbstractVecOrMat{<:Base.HWReal}, A::AbstractMatrix{<:Base.HWReal}, B::AbstractVecOrMat{<:Base.HWReal}, ::False, ::True) where {F}
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = f(Cmn)
  end
end
function dense!(f::F, C::AbstractVecOrMat{<:Base.HWReal}, A::AbstractMatrix{<:Base.HWReal}, B::AbstractVecOrMat{<:Base.HWReal}, ::False, ::False) where {F}
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn
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
  ∂fx = getfield(ForwardDiff.partials(dx).values,1)
  fx, ∂fx
end
# overloadable
@inline ∂(f::F) where {F} = ForwardDiffElementwise{F}(f)

function get∂C(td::TurboDense{B,D}, C::AbstractArray, ∂Cp::Ptr{UInt8}) where {B,D}
  get∂C(td, C, ∂Cp, fast_fuse(td))
end
function get∂C(td::TurboDense, C::AbstractArray{T}, ∂Cp::Ptr{UInt8}, ::True) where {T}
  ∂C = PtrArray(reinterpret(Ptr{T}, ∂Cp), size(C))
  ∂Cp += align(length(∂C)*sizeof(T))
  ∂C, ∂Cp
end
function get∂C(td::TurboDense, C::AbstractArray{T}, ∂Cp::Ptr{UInt8}, ::False) where {T}
  lenC = length(C)
  ∂C = PtrArray(reinterpret(Ptr{T}, ∂Cp), (lenC,))
  ∂Cp += align(lenC*sizeof(T))
  ∂C, ∂Cp
end
function get∂C(td::TurboDense{B,D,typeof(relu)}, C::AbstractArray, ∂Cp::Ptr{UInt8}) where {B,D}
  ∂C = PtrArray(reinterpret(Ptr{Bit}, ∂Cp), size(C))
  ∂Cp += align((length(∂C) + 7) >>> 3)
  ∂C, ∂Cp
end
get∂C(td::TurboDense{B,D,typeof(identity)}, C::AbstractArray, ∂Cp::Ptr{UInt8}) where {B,D} = (nothing, ∂Cp)

# generic
function dense!(f::F, ∂C::AbstractArray{T1,N}, C::AbstractArray{T2,N}, A::AbstractMatrix, B::AbstractArray{T3,N}, ::True) where {F,T1,T2,T3,N}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  ∂f = ∂(f)
  @turbo for n ∈ indices((B,C,∂C),2), m ∈ indices((A,C,∂C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    Cmn, ∂Cmn = ∂f(Cmn + A[m,Kp1])
    ∂C[m,n] = ∂Cmn
    C[m,n] = Cmn
  end
end
function dense!(f::F, ∂C::AbstractVector, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, ::True) where {F}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn + A[m,Kp1]
  end
  ∂f = ∂(f)
  @turbo for i ∈ eachindex(C)
    Cᵢ, ∂Cᵢ = ∂f(C[i])
    ∂C[i] = ∂Cᵢ
    C[i] = Cᵢ
  end
end

function dense!(f::F, ∂C::AbstractArray{T1,N}, C::AbstractArray{T2,N}, A::AbstractMatrix, B::AbstractArray{T3,N}, ::False) where {F,T1,T2,T3,N}
  ∂f = ∂(f)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    Cmn, ∂Cmn = ∂f(Cmn)
    C[m,n] = Cmn
    ∂C[m,n] = ∂Cmn
  end
end
function dense!(f::F, ∂C::AbstractVector, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, ::False) where {F}
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn
  end
  ∂f = ∂(f)
  @turbo for i ∈ eachindex(C)
    Cᵢ, ∂Cᵢ = ∂f(C[i])
    C[i] = Cᵢ
    ∂C[i] = ∂Cᵢ
  end
end

function dense!(::typeof(tanh), ∂C::AbstractVector, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, ::True)
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn + A[m,Kp1]
  end
  @turbo for i ∈ eachindex(C)
    Cᵢ = tanh(C[i])
    C[i] = Cᵢ
    ∂C[i] = one(Cᵢ) - Cᵢ*Cᵢ
  end
end
function dense!(f::typeof(tanh), ∂C::AbstractVector, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, ::False)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn
  end
  @turbo for i ∈ eachindex(C)
    Cᵢ = tanh(C[i])
    C[i] = Cᵢ
    ∂C[i] = one(Cᵢ) - Cᵢ*Cᵢ
  end
end
function dense!(f::typeof(relu), ∂C::AbstractArray{Bit,N}, C::AbstractArray{T1,N}, A::AbstractMatrix, B::AbstractArray{T2,N}, ::True) where {T1,T2,N}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    Cmnr = Cmn + A[m,Kp1]
    Cmnr_gt_0 = Cmnr > zero(Cmnr)
    C[m,n] = ifelse(Cmnr_gt_0, Cmnr, zero(Cmnr))
    ∂C[m,n] = Cmnr_gt_0
  end
end
function dense!(f::typeof(relu), ∂C::AbstractArray{Bit,N}, C::AbstractArray{T1,N}, A::AbstractMatrix, B::AbstractArray{T2,N}, ::False) where {T1,T2,N}
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    Cmn_gt_0 = Cmn > zero(Cmn)
    C[m,n] = ifelse(Cmn_gt_0, Cmn, zero(Cmn))
    ∂C[m,n] = Cmn_gt_0
  end
end
function dense!(f::typeof(identity), ::Nothing, C::AbstractArray{T1,N}, A::AbstractMatrix, B::AbstractArray{T2,N}, ::True) where {T1,T2,N}
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn + A[m,Kp1]
  end
end
function dense!(f::typeof(identity), ::Nothing, C::AbstractArray{T1,N}, A::AbstractMatrix, B::AbstractArray{T2,N}, ::False) where {T1,T2,N}
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn
  end
end

# struct DensePullBack{B,D,F,T,AT}
#   td::TurboDense{B,D,F}
#   p::Ptr{T}
#   A::AT
# end





function valgrad_layer!(pg::Ptr{T}, td::TurboDense{O}, B, p::Ptr{T}, pu::Ptr{UInt8}) where {T,O}
  batch_size = size(B, StaticInt(2))
  pu2 = Base.unsafe_convert(Ptr{T}, pu + batch_size * getfield(td.dims,2) * sizeof(T))
  C, _pu3 = alloc_return(td, batch_size, pu2, contiguous_axis(B), stride_rank(B))
  pu3 = Base.unsafe_convert(Ptr{UInt8}, _pu3)
  ∂C, _ = get∂C(td, C, pu)
  A, p2 = getparams(td, p)
  f = td.f
  dense!(f, ∂C, C, A, B, static(O))
  # doesn'tneed a pullback
  pg + length(A)*sizeof(T), C, p2, pu3
end
alloc_return_B_dense(B::PtrArray, pu::Ptr{UInt8}, _) = (B, pu) # assume `PtrArray` means we can destroy it
function alloc_return_B_dense(B::AbstractArray{T}, pu::Ptr{UInt8}, input_dim) where {T}
  si = bytestrideindex(B)
  sp = stridedpointer(reinterpret(Ptr{T}, pu), si)
  B̄ = PtrArray(sp, (input_dim, size(B, static(2))), StrideArraysCore.val_dense_dims(B))
  B̄, pu + align(length(B̄)*sizeof(T))
end
function pullback!(pg::Ptr{T}, td::TurboDense{O}, _C̄, B, p::Ptr{T}, pu::Ptr{UInt8}, pu2::Ptr{UInt8}) where {T,O}
  # Start with 4-arg `pulback!` to update `∂C`
  C̄ = pullback_param!(pg, td, _C̄, B, p, pu) # Ā = C̄ * B'
  # Now 5-arg
  # B̄ = A' * C̄
  A, _  = getparams(td, p)
  B̄, pu2 = alloc_return_B_dense(B, pu2, input_dims(td))
  dense!(identity, nothing, B̄, matrix_view(td, A)', C̄, False())
  B̄, pu2
end
matrix_view(::TurboDense{false}, A) = A
function matrix_view(::TurboDense{true}, A)
  Kp1 = ArrayInterface.size(A, StaticInt(2))
  K = Kp1 - StaticInt(1)
  view(A, :, static(1):K)
end
upate_C̄!(C̄, ∂C, td::TurboDense{B,D,typeof(identity)}) where {B,D} = nothing
function upate_C̄!(C̄, ∂C, td::TurboDense{B,D}) where {B,D}
  @turbo for i ∈ eachindex(∂C)
    C̄[i] *= ∂C[i]
  end
end
function pullback_param!(pg::Ptr{T}, td::TurboDense{O}, C̄, B, p::Ptr{T}, pu::Ptr{UInt8}) where {T,O}
  # Ā = C̄ * B'
  ∂C, pu2 = get∂C(td, C̄, pu)
  upate_C̄!(C̄, ∂C, td)
  Ā, _  = getparams(td, pg)
  dense_param_update!(td, Ā, C̄, B)
  C̄
end
function dense_param_update!(::TurboDense{true}, Ā, C̄, B)
  Kp1 = ArrayInterface.size(Ā, StaticInt(2))
  K = Kp1 - StaticInt(1)
  dense!(identity, nothing, view(Ā, :, static(1):K), C̄, B', False())
  @turbo for m ∈ axes(Ā,1)
    s = zero(eltype(Ā))
    for n ∈ axes(C̄,2)
      s += C̄[m,n]
    end
    Ā[m,Kp1] = s
  end
end
dense_param_update!(::TurboDense{false}, Ā, C̄, B) = dense!(identity, nothing, Ā, C̄, B', False())



struct DualCall{F}; f::F; end
@generated function (dc::DualCall)(x::T, y::Vararg{T,P}) where {P,T}
  quote
    $(Expr(:meta,:inline))
    @inbounds d = Base.Cartesian.@ncall $P ForwardDiff.Dual x p -> y[p]
    fd = dc.f(d)
    ∂fd = ForwardDiff.partials(fd)
    @inbounds (ForwardDiff.value(fd), (Base.Cartesian.@ntuple $P p -> ∂fd[p])...)
  end
end
dualeval!(f::typeof(identity), Cdual::AbstractArray{D}) where {T<:Union{Float32,Float64}, P, D<:ForwardDiff.Dual{<:Any,T,P}} = nothing
@generated function dualeval!(f::F, Cdual::AbstractArray{D}) where {F,T<:Union{Float32,Float64}, P, D<:ForwardDiff.Dual{<:Any,T,P}}
  quote
    C = reinterpret(reshape, T, Cdual)
    g = DualCall(f)
    @turbo for m ∈ eachindex(Cdual)
      (Base.Cartesian.@ntuple $(P+1) p -> C[p,m]) = Base.Cartesian.@ncall $(P+1) g p -> C[p,m]
    end
  end
end

# maybe collapses dims 1 and 2 of a 3d array.
collapse_dims12(::AbstractArray, A) = A
function collapse_dims12(A::PtrArray{S,(true,true),T,1,1,0,(1,2)}) where {S,T}
  M, N = size(A)
  sp = stridedpointer(A)
  o1, o2 = offsets(sp)
  x1, x2 = strides(sp)
  si = ArrayInterface.StrideIndex{2,(1,2),1}((x1,StaticInt(0)), (o1, o2))
  spnew = stridedpointer(pointer(sp), si)
  PtrArray(sp, (M*N,StaticInt(1)), Val((true,true)))
end
function collapse_dims12(A::PtrArray{S,(true,true,true),T,3,1,0,(1,2,3)}) where {S,T}
  M, N, P = size(A)
  sp = stridedpointer(A)
  o1, o2, o3 = offsets(sp)
  x1, x2, x3 = strides(sp)
  si = ArrayInterface.StrideIndex{3,(1,2,3),1}((x1,StaticInt(0),x3), (o1, o2, o3))
  spnew = stridedpointer(pointer(sp), si)
  PtrArray(sp, (M*N, StaticInt(1), P), Val((true,true,true)))
end
const Collapsible12PtrArray{S,T} = Union{PtrArray{S,(true,true),T,1,1,0,(1,2)},PtrArray{S,(true,true,true),T,3,1,0,(1,2,3)}}
function collapse_dims12(A::Collapsible12PtrArray, O::AbstractArray)
  StrideArray(collapse_dims12(A), O)
end
function collapse_dims12(A::AbstractArray)
  collapse_dims12(PtrArray(A), A)
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

function matmul!(Cdual::AbstractVector{D}, Adual::AbstractMatrix{D}, B::AbstractVector{T}, ::True) where {T<:Union{Float32,Float64},P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  Kp1 = ArrayInterface.size(A, StaticInt(3))
  K = Kp1 - StaticInt(1)
  # Pp1 = StaticInt(P) + StaticInt(1)
  Af, Cf = collapse_dims12(A, C)
  @turbo for m ∈ indices((Af,Cf),2), p ∈ indices((Af,Cf),1)
    Cmn = zero(T)
    for k ∈ 1:K
      Cmn += Af[p,m,k] * B[k]
    end
    Cf[p,m] = Cmn + Af[p, m, Kp1]
  end
end
function matmul!(Cdual::AbstractMatrix{D}, Adual::AbstractMatrix{D}, B::AbstractMatrix{T}, ::True) where {T<:Union{Float32,Float64},P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  Kp1 = ArrayInterface.size(A, StaticInt(3))
  K = Kp1 - StaticInt(1)
  # Pp1 = StaticInt(P) + StaticInt(1)
  Af, Cf = collapse_dims12(A, C)
  @turbo for n ∈ indices((B,Cf),(2,3)), m ∈ indices((Af,Cf),2), p ∈ indices((Af,Cf),1)
    Cmn = zero(T)
    for k ∈ 1:K
      Cmn += Af[p,m,k] * B[k,n]
    end
    Cf[p, m, n] = Cmn + Af[p, m, Kp1]
  end
end
function matmul!(Cdual::AbstractVector{D}, Adual::AbstractMatrix{D}, B::AbstractVector{T}, ::False) where {T<:Union{Float32,Float64},P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  # Pp1 = StaticInt(P) + StaticInt(1)
  Af, Cf = collapse_dims12(A, C)
  @turbo for m ∈ indices((Af,Cf),2), p ∈ indices((Af,Cf),1)
    Cmn = zero(T)
    for k ∈ indices((Af,B),(3,1))
      Cmn += Af[p,m,k] * B[k]
    end
    Cf[p,m] = Cmn
  end
end
function matmul!(Cdual::AbstractMatrix{D}, Adual::AbstractMatrix{D}, B::AbstractMatrix{T}, ::False) where {T<:Union{Float32,Float64},P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  # Pp1 = StaticInt(P) + StaticInt(1)
  Af, Cf = collapse_dims12(A, C)
  @turbo for n ∈ indices((B,Cf),(2,3)), m ∈ indices((Af,Cf),2), p ∈ indices((Af,Cf),1)
    Cmn = zero(T)
    for k ∈ indices((Af,B),(3,1))
      Cmn += Af[p,m,k] * B[k,n]
    end
    Cf[p,m,n] = Cmn
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
function matmul!(Cdual::AbstractVector{D}, Adual::AbstractMatrix{D}, Bdual::AbstractVector{D}, ::True) where {T<:Union{Float32,Float64},P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  B = reinterpret(reshape, T, Bdual)
  Kp1 = ArrayInterface.size(A, StaticInt(3))
  K = Kp1 - StaticInt(1)
  Pstatic = StaticInt(P)
  Af, Cf1 = collapse_dims12(A, C)
  @turbo for m ∈ indices((Af,Cf1),2), p ∈ indices((Af,Cf1),1)
    Cmn = zero(T)
    for k ∈ 1:K
      Cmn += Af[p, m, k] * B[1, k]
    end
    Cf1[p, m] = Cmn + Af[p, m, Kp1]
  end
  @turbo for n ∈ indices((B,C),3), m ∈ indices((A,C),2), p ∈ 1:Pstatic, k ∈ 1:K
    C[p+1, m] += A[1, m, k] * B[p+1, k]
  end
end
function matmul!(Cdual::AbstractMatrix{D}, Adual::AbstractMatrix{D}, Bdual::AbstractMatrix{D}, ::True) where {T<:Union{Float32,Float64},P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  B = reinterpret(reshape, T, Bdual)
  Kp1 = ArrayInterface.size(A, StaticInt(3))
  K = Kp1 - StaticInt(1)
  Pstatic = StaticInt(P)
  Af, Cf1 = collapse_dims12(A, C)
  @turbo for n ∈ indices((B,Cf1),3), m ∈ indices((Af,Cf1),2), p ∈ indices((Af,Cf1),1)
    Cmn = zero(T)
    for k ∈ 1:K
      Cmn += Af[p, m, k] * B[1, k, n]
    end
    Cf1[p, m, n] = Cmn + Af[p, m, Kp1]
  end
  # Bf, Cf2 = collapse_dims12(clip_r1_view(B), clip_r1_view(C))
  @turbo for n ∈ indices((B,C),3), m ∈ indices((A,C),2), p ∈ 1:Pstatic, k ∈ 1:K
    C[p+1,m,n] += A[1,m,k] * B[p+1,k,n]
  end
end
function matmul!(Cdual::AbstractVector{D}, Adual::AbstractMatrix{D}, Bdual::AbstractVector{D}, ::False) where {T<:Union{Float32,Float64},P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  B = reinterpret(reshape, T, Bdual)
  Pstatic = StaticInt(P)
  Af, Cf1 = collapse_dims12(A, C)
  @turbo for m ∈ indices((Af,Cf1),2), p ∈ indices((Af,Cf1),1)
    Cmn = zero(T)
    for k ∈ indices((Af,B),(3,2))
      Cmn += Af[p,m,k] * B[1,k]
    end
    Cf1[p,m] = Cmn
  end
  @turbo for n ∈ indices((B,C),3), m ∈ indices((A,C),2), p ∈ 1:Pstatic, k ∈ indices((A,B),(3,2))
    C[p+1,m] += A[1,m,k] * B[p+1,k]
  end
end


function matmul!(Cdual::AbstractMatrix{D}, Adual::AbstractMatrix{D}, Bdual::AbstractMatrix{D}, ::False) where {T<:Union{Float32,Float64},P,D<:ForwardDiff.Dual{<:Any,T,P}}
  C = reinterpret(reshape, T, Cdual)
  A = reinterpret(reshape, T, Adual)
  B = reinterpret(reshape, T, Bdual)
  Pstatic = StaticInt(P)
  Af, Cf1 = collapse_dims12(A, C)
  @turbo for n ∈ indices((B,Cf1),3), m ∈ indices((Af,Cf1),2), p ∈ indices((Af,Cf1),1)
    Cmn = zero(T)
    for k ∈ indices((Af,B),(3,2))
      Cmn += Af[p,m,k] * B[1,k,n]
    end
    Cf1[p,m,n] = Cmn  
  end
  # Bf, Cf2 = collapse_dims12(clip_r1_view(B), clip_r1_view(C))
  @turbo for n ∈ indices((B,C),3), m ∈ indices((A,C),2), p ∈ 1:Pstatic, k ∈ indices((A,B),(3,2))
    C[p+1,m,n] += A[1,m,k] * B[p+1,k,n]
  end
end

function dense!(f::F, Cdual::AbstractArray{D}, Adual::AbstractMatrix{D}, B::AbstractArray, ::BT, ::FF) where {F,BT,FF,T<:Union{Float32,Float64},P,D<:ForwardDiff.Dual{<:Any,T,P}}
  matmul!(Cdual, Adual, B, BT())
  dualeval!(f, Cdual)
end
# derivatives

@generated function Base.abs(x::ForwardDiff.Dual{TAG,S,N}) where {TAG,S<:AbstractSIMD,N}
  quote
    $(Expr(:meta,:inline))
    val = x.value
    p = x.partials
    cmp = val < zero($S)
    absx = $ifelse(cmp, -val, val)
    Base.Cartesian.@nexprs $N n -> p_n = p[n]
    ForwardDiff.Dual{$TAG}(absx, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> $ifelse(cmp, -p_n, p_n)))
  end
end
@inline function Base.max(
  x::ForwardDiff.Dual{TAG,<:AbstractSIMD,N},
  y::ForwardDiff.Dual{TAG,<:AbstractSIMD,N}
) where {TAG,N}
  vx = ForwardDiff.value(x)
  vy = ForwardDiff.value(y)
  xgy = vx > vy
  z = LoopVectorization.ifelse(xgy, vx, vy)
  p = let px = ForwardDiff.partials(x), py = ForwardDiff.partials(y), xgy = xgy
    ntuple(Val(N)) do n
      LoopVectorization.ifelse(xgy, px[n], py[n])
    end
  end
  ForwardDiff.Dual{TAG}(z, p...)
end

@inline Base.max(x::T, y::Real) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
@inline Base.max(y::Real, x::T) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
@inline Base.max(x::T, y::Int) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
@inline Base.max(y::Int, x::T) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
