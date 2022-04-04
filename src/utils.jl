
tsprod(x) = ArrayInterface.reduce_tup(*, x)
tsprod(::Tuple{}) = static(1)

function maximum_turbo!(m, y)
  @turbo for i in eachindex(m)
    mi = typemin(eltype(y))
    for j in axes(y, 1)
      mi = max(mi, y[j, i])
    end
    m[i] = mi
  end
end


function unnormalized_logsoftmax!(z, m, y::AbstractMatrix)
  maximum_turbo!(m, y)
  @turbo for j in axes(y, 2)
    mj = m[j]
    s = zero(eltype(m))
    for i in axes(y, 1)
      yij = y[i, j]
      zij = mj == Inf ? (yij == Inf ? zero(eltype(y)) : -Inf) : yij - m[j]
      z[i, j] = zij
      s += exp(zij)
    end
    m[j] = s
  end
  @turbo for i in eachindex(m)
    m[i] = log(m[i])
  end
end
function logsoftmax!(z, m, y::AbstractMatrix)
  unnormalized_logsoftmax!(z, m, y)
  @turbo for j in axes(z, 2), i in axes(z, 1)
    z[i, j] -= m[j]
  end
end
function logsoftmax(y::AbstractMatrix)
  m = similar(y, size(y, 2))
  z = similar(y)
  logsoftmax!(z, m, y)
  return z
end
