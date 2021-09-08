using VectorizedRNG

struct Dropout{R <: Union{Nothing,AbstractRNG}}
  p::Float64
  rng::R
end
getrng(d::Dropout{Nothing}) = local_rng()
getrng(d::Dropout{<:AbstractRNG}) = getfield(d, :rng)

gradval(d::Dropout) = inv(1.0 - p)

function (d::Dropout)(B::AbstractVecOrMat{T}, p::Ptr, pu::Ptr) where {T}
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



