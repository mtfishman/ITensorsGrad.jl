using ITensors
using NDTensors
using LinearAlgebra
using Zygote
using ChainRulesCore

import Base: +
import ChainRulesCore: rrule

#
# NDTensors rrules
#

function rrule(::Type{<:Dense}, data::AbstractVector)
  function Dense_pullback(ȳ)
    return (NO_FIELDS, ȳ)
  end
  return Dense(data), Dense_pullback
end

function rrule(::Type{<:Diag}, data)
  function Diag_pullback(ȳ)
    return (NO_FIELDS, ȳ)
  end
  return Diag(data), Diag_pullback
end

function rrule(::Type{<:Tensor}, is::IndexSet, store::TensorStorage)
  function Tensor_pullback(ȳ)
    return (NO_FIELDS, DoesNotExist(), ȳ)
  end
  return tensor(store, is), Tensor_pullback
end

# TODO: may not be needed
function rrule(::typeof(mapprime), A, B) 
  function mapprime_pullback(ȳ)
    return (NO_FIELDS, ȳ, DoesNotExist())
  end
  return mapprime(A, B), mapprime_pullback
end

#
# ITensors rrules
#

function rrule(::Type{<:Index}, x::Vararg{<:Any, N}) where {N}
  function Index_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return Index(x...), Index_pullback
end

function rrule(::Type{<:IndexSet}, x::Vararg{<:Any, N}) where {N}
  function IndexSet_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return IndexSet(x...), IndexSet_pullback
end

function rrule(::Type{<:ITensor}, is::IndexSet, store::TensorStorage)
  function ITensor_pullback(ȳ::Base.RefValue)
    println()
    println("In ITensor_pullback ::RefValue")
    @show ȳ[]
    @show typeof(ȳ[].store)
    @show typeof(ȳ[].inds)
    return (NO_FIELDS, DoesNotExist(), ȳ[].store.store)
  end
  function ITensor_pullback(ȳ::ITensor)
    println()
    println("In ITensor_pullback ::ITensor")
    @show ȳ
    @show typeof(ȳ)
    return (NO_FIELDS, DoesNotExist(), ȳ.store)
  end
  function ITensor_pullback(ȳ::Tensor)
    println()
    println("In ITensor_pullback ::Tensor")
    @show ȳ
    @show typeof(ȳ)
    return (NO_FIELDS, DoesNotExist(), ȳ.store)
  end
  function ITensor_pullback(ȳ::AbstractArray)
    println()
    println("In ITensor_pullback ::AbstractArray")
    @show ȳ
    @show typeof(ȳ)
    return (NO_FIELDS, DoesNotExist(), ȳ)
  end
  return itensor(store, is), ITensor_pullback
end

function rrule(::typeof(*), T1::ITensor, T2::ITensor)
  function times_pullback(ȳ::ITensor)
    return (NO_FIELDS, ȳ * T2, T1 * ȳ)
  end
  function times_pullback(ȳ::Base.RefValue{Any})
    @show typeof(ȳ[])
    if ȳ[].store isa Union{Tensor, AbstractArray}
      T̄ = itensor(ȳ[].store)
    elseif ȳ[].store isa TensorStorage && ȳ[].inds isa IndexSet
      T̄ = itensor(ȳ[].store, ȳ[].inds)
    else
      error("No times_pullback defined")
    end
    return (NO_FIELDS, T̄ * T2, T1 * T̄)
  end
  return T1 * T2, times_pullback
end

# To get around Zygote keyword argument issue
import ITensors: dag
function dag(T::ITensor)
  @show typeof(T)
  TT = conj(tensor(T))
  return itensor(store(TT), dag(inds(T)))
end

function rrule(::typeof(dag), T::ITensor)
  function dag_pullback(ȳ::Base.RefValue)
    println()
    println("In dag_pullback ::RefValue")
    @show ȳ
    @show typeof(ȳ)
    return (NO_FIELDS, conj(ȳ[].store))
  end
  function dag_pullback(ȳ::ITensor)
    println()
    println("In dag_pullback ::ITensor")
    @show ȳ
    @show typeof(ȳ)
    return (NO_FIELDS, ȳ)
  end
  return dag(T), dag_pullback
end

# TODO: are these needed?
function (A::Base.RefValue{Any} + B::ITensor)
  @show typeof(A[].store)
  @show typeof(A[].inds)
  return itensor(A[].store, inds(B)) + B
end

# TODO: are these needed?
function (A::ITensor + B::Base.RefValue{Any})
  @show typeof(B[].store)
  @show typeof(B[].inds)
  return A + itensor(B[].store, inds(A))
end

# TODO: define this
#+(::ITensor{2}, ::ITensor{0})

T(β) = [ exp(β) exp(-β)
        exp(-β)  exp(β)]

Z(β, N) = tr(T(β)^N)

N = 2
Z(β) = Z(β, N)

β = 0.1
function Zit(β)
  #i = Index(2, "i")
  #A = itensor(T(β), i', dag(i))
  #return (A * dag(A))[]
  
  i = Index(2, "i")
  A = itensor(T(β), i', dag(i))
  #return tr(product(A, A))
  #return (prime(A) * A * δ(dag(i)'', i))[]

  A2 = mapprime(A' * A, 2 => 1)
  return (A2 * δ(dag(i)', i))[]

  #i = Index(1, "i")
  #A = itensor([β^2], i', dag(i))
  #return dag(A)[1, 1]
  #return A[1, 1]
end

@show Z(β)
@show Zit(β)
println()

@show Z'(β)
@show Zit'(β)
println()


