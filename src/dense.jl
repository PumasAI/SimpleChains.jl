
struct TurboDense{B,D,F}
  f::F
  dims::D
end

numparams(d::TurboDense{false}) = prod(d.dims)
function numparams(d::TurboDense{true})
  id, od = d.dims
  id * od + od
end
function output_size(::Val{T}, td::TurboDense, batch_size) where {T}
  g1 = numparams(td) # for gradients
  g2 = getfield(d.dims, 1) * batch_size # for output
  align(static_sizeof(T) * g1) + align(static_sizeof(T) * g2)
end

fast_fuse(::typeof(relu)) = True()
fast_fuse(::typeof(abs)) = True()
fast_fuse(::typeof(abs2)) = True()
fast_fuse(::typeof(identity)) = True()
fast_fuse(_) = False()

function getparams(td::TurboDense{true}, p::Ptr{UInt8}, ::Val{T}) where {T}
  id, od = td.dims
  PtrArray(reinterpret(Ptr{T}, p), (id, od)), p + id * od * sizeof(T)
end
function getparams(td::TurboDense{true}, p::Ptr{UInt8}, ::Val{T}) where {T}
  id, od = td.dims
  idp1 = id + StaticInt(1)
  W = PtrArray(reinterpret(Ptr{T}, p), (od, idp1))
  W, p + (id * odp1) * sizeof(T)
end

function alloc_return(td::TurboDense, batch_size, p::Ptr{T}) where {T}
  O = getfield(td.dims,2)
  PtrArray(p, (O, batch_size)), p + align(O*batch_size*sizeof(T))
end


function (td::TurboDense{O})(B, p::Ptr{T}, pu::Ptr{UInt8}) where {T,O}
  C, p = alloc_return(td, size(B, StaticInt(2)), p)
  A, pu = getparams(td, pu, Val(T))
  f = td.f
  dense!(f, C, A, B, static(O), fast_fuse(f)), p, pu
end


function dense!(f::F, C, A, B, ::True, ::True) where {F}
  K = ArrayInterface.size(A, StaticInt(2))
  Kp1 = K + StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = f(Cmn + A[m,Kp1])
  end
end
function dense!(f::F, C, A, B, ::True, ::False) where {F}
  K = ArrayInterface.size(A, StaticInt(2))
  Kp1 = K + StaticInt(1)
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

function dense!(f::F, C, A, B, ::False, ::True) where {F}
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = f(Cmn)
  end
end
function dense!(f::F, C, A, B, ::False, ::False) where {F}
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
@inline function (fw::ForwardDiffElementwise{F})(x)
  dx = fw.f(ForwardDiff.Dual(x, one(x)))
  fx = ForwardDiff.value(dx)
  ∂fx = getfield(ForwardDiff.partials(dx).values,1)
  fx, ∂fx
end
# overloadable
@inline ∂(f::F) where {F} = ForwardDiffElementwise{F}(f)



# generic
function dense!(f::F, ∂Cp, C::AbstractArray{T}, A, B, ::True, ::True) where {F,T}
  K = ArrayInterface.size(A, StaticInt(2))
  Kp1 = K + StaticInt(1)
  ∂C = PtrArray{T}(∂Cp, size(C))
  ∂f = ∂(f)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    Cmn, ∂Cmn = ∂f(Cmn + A[m,Kp1])
    C[m,n] = Cmn
    ∂C[m,n] = ∂Cmn    
  end
  ∂Cp + align(length(∂C)*sizeof(T))
end
function dense!(f::F, ∂Cp, C, A, B, ::True, ::False) where {F}
  K = ArrayInterface.size(A, StaticInt(2))
  Kp1 = K + StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn + A[m,Kp1]
  end
  ∂f = ∂(f)
  lenC = length(C)
  ∂C = PtrArray{T}(∂Cp, (lenC,))
  @turbo for i ∈ eachindex(C)
    Cᵢ, ∂Cᵢ = ∂f(C[i])
    C[i] = Cᵢ
    ∂C[i] = ∂Cᵢ    
  end
  ∂Cp + align(lenC*sizeof(T))
end

function dense!(f::F, ∂Cp, C, A, B, ::False, ::True) where {F}
  ∂f = ∂(f)
  ∂C = PtrArray{T}(∂Cp, size(C))
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    Cmn, ∂Cmn = ∂f(Cmn)
    C[m,n] = Cmn
    ∂C[m,n] = ∂Cmn    
  end
  ∂Cp + align(length(∂C)*sizeof(T))
end
function dense!(f::F, ∂Cp, C, A, B, ::False, ::False) where {F}
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn
  end
  ∂f = ∂(f)
  lenC = length(C)
  ∂C = PtrArray{T}(∂Cp, (lenC,))
  @turbo for i ∈ eachindex(C)
    Cᵢ, ∂Cᵢ = ∂f(C[i])
    C[i] = Cᵢ
    ∂C[i] = ∂Cᵢ
  end
  ∂Cp + align(lenC*sizeof(T))
end

function dense!(::typeof(tanh), ∂Cp, C, A, B, ::True, ::False)
  K = ArrayInterface.size(A, StaticInt(2))
  Kp1 = K + StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn + A[m,Kp1]
  end
  lenC = length(C)
  ∂C = PtrArray{T}(∂Cp, (lenC,))
  @turbo for i ∈ eachindex(C)
    Cᵢ = tanh(C[i])
    C[i] = Cᵢ
    ∂C[i] = Cᵢ * (one(Cᵢ) - Cᵢ)
  end
  ∂Cp + align(lenC*sizeof(T))
end

function dense!(f::typeof(tanh), ∂Cp, C, A, B, ::False, ::False)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn
  end
  lenC = length(C)
  ∂C = PtrArray{T}(∂Cp, (lenC,))
  @turbo for i ∈ eachindex(C)
    Cᵢ = tanh(C[i])
    C[i] = Cᵢ
    ∂C[i] = Cᵢ * (one(Cᵢ) - Cᵢ)
  end
  ∂Cp + align(lenC*sizeof(T))
end
function dense!(f::typeof(relu), ∂Cp, C, A, B, ::True, ::True)
  K = ArrayInterface.size(A, StaticInt(2))
  Kp1 = K + StaticInt(1)
  ∂C = PtrArray{Bit}(reinterpret(Ptr{Bit}, ∂Cp), size(C))
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
  ∂Cp + align((length(∂C) + 7) >>> 3)
end
function dense!(f::typeof(relu), ∂Cp, C, A, B, ::False, ::True)
  ∂C = PtrArray{Bit}(reinterpret(Ptr{Bit}, ∂Cp), size(C))
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    Cmnr = Cmn + A[m,Kp1]
    Cmnr_gt_0 = Cmnr > zero(Cmnr)
    C[m,n] = ifelse(Cmnr_gt_0, Cmnr, zero(Cmnr))
    ∂C[m,n] = Cmnr_gt_0
  end
  ∂Cp + align((length(∂C) + 7) >>> 3)
end
function dense!(f::typeof(identity), ∂Cp, C, A, B, ::True, ::True)
  K = ArrayInterface.size(A, StaticInt(2))
  Kp1 = K + StaticInt(1)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ 1:K
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn + A[m,Kp1]
  end
  ∂Cp
end
function dense!(f::typeof(identity), ∂Cp, C, A, B, ::False, ::True)
  @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
    Cmn = zero(eltype(C))
    for k ∈ indices((A,B),(2,1))
      Cmn += A[m,k] * B[k,n]
    end
    C[m,n] = Cmn
  end
  ∂Cp
end

# struct DensePullBack{B,D,F,T,AT}
#   td::TurboDense{B,D,F}
#   p::Ptr{T}
#   A::AT
# end





function valgrad_layer!(pg::Ptr{T}, td::TurboDense{O}, B, p::Ptr{T}, pu::Ptr{UInt8}) where {T,O}
  C, pu2 = alloc_return(td, size(B, StaticInt(2)), pu)
  A, p2 = getparams(td, p, Val(T))
  f = td.f
  pu3 = dense!(f, pu2, C, A, B, static(O), fast_fuse(f))
  # doesn'tneed a pullback
  pg + length(p2)*sizeof(T), C, nothing, p2, pu3
end

function pullback(td::TurboDense{O}, B, p::Ptr{T}, pu::Ptr{UInt8}) where {T,O}
  # Ā = C̄ * B'
  # B̄ = A' * C̄
end

