using LinearAlgebra
using ChainRulesCore
using Zygote

import Base: getindex, size, *, ^, convert
import LinearAlgebra: tr
import ChainRulesCore: rrule

#
# Type definition
#

struct MyArray{T, N}
  v::Vector{T}
  t::NTuple{N, Int}
end
MyArray(A::Array) = MyArray(vec(A), size(A))

convert(::Type{<:Array}, M::MyArray) = reshape(M.v, M.t)

getindex(M::MyArray, args...) = getindex(convert(Array, M), args...)
size(M::MyArray, n::Int) = M.t[n]
tr(M::MyArray) = tr(convert(Array, M))

(M1::MyArray{Float64,2} * M2::MyArray{Float64,2}) =
  convert(Array, M1) * convert(Array, M2)

(M::MyArray{Float64,2} ^ n::Int) =
  convert(Array, M) ^ n

#
# ChainRules definitions
#

function rrule(::Type{<:MyArray}, v::Vector, t::NTuple)
  function MyArray_pullback(ȳ::AbstractArray)
    return (NO_FIELDS, ȳ, DoesNotExist())
  end
  function MyArray_pullback(ȳ::Composite)
    return (NO_FIELDS, ȳ.v, DoesNotExist())
  end
  return MyArray(v, t), MyArray_pullback
end

x = 3.0

A(x) = [exp(x) exp(-x); exp(-x) exp(x)]
M(x) = MyArray(A(x))

fA(x) = tr(A(x))
fM(x) = tr(M(x))

@show fA(x)
@show fM(x)
@show fA'(x)
@show fM'(x)
println()

fA(x) = A(x)[1, 1]
fM(x) = M(x)[1, 1]

@show fA(x)
@show fM(x)
@show fA'(x)
@show fM'(x)
println()

fA(x) = tr(A(x) * A(x))
fM(x) = tr(M(x) * M(x))

@show fA(x)
@show fM(x)
@show fA'(x)
@show fM'(x)
println()

N = 5
fA(x) = tr(A(x) ^ N)
fM(x) = tr(M(x) ^ N)

@show fA(x)
@show fM(x)
@show fA'(x)
@show fM'(x)
println()


