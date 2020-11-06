#
# NDTensors rrules
#

function rrule(::Type{<:Dense}, v::AbstractVector)

  function Dense_pullback(ΔΩ)
    return (NO_FIELDS, data(ΔΩ))
  end

  return Dense(v), Dense_pullback
end

function rrule(::Type{<:Combiner}, x::Vararg{<:Any, N}) where {N}
  function Combiner_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return Combiner(x...), Combiner_pullback
end

function rrule(::Type{<:Diag}, v)

  function Diag_pullback(ΔΩ)
    return (NO_FIELDS, data(ΔΩ))
  end

  function Diag_pullback(ΔΩ::ITensor)
    return (NO_FIELDS, data(store(ΔΩ)))
  end

  return Diag(v), Diag_pullback
end

# XXX Not sure about this definition
function rrule(::Type{<:Tensor},
               is::IndexSet, st::TensorStorage)

  function Tensor_pullback(ΔΩ)
    #return (NO_FIELDS, DoesNotExist(), itensor(store(ΔΩ), is))
    return (NO_FIELDS, DoesNotExist(), store(ΔΩ))
  end

  return tensor(st, is), Tensor_pullback
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

function rrule(::typeof(unioninds), x::Vararg{<:Any, N}) where {N}
  function unioninds_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return unioninds(x...), unioninds_pullback
end

function rrule(::typeof(commoninds), x::Vararg{<:Any, N}) where {N}
  function commoninds_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return commoninds(x...), commoninds_pullback
end

function rrule(::Type{<:ITensor},
               is::IndexSet, st::TensorStorage)

  function ITensor_pullback(ΔΩ::TensorStorage)
    return (NO_FIELDS, DoesNotExist(), ΔΩ)
  end

  # TODO: maybe return Dense(ΔΩ)?
  function ITensor_pullback(ΔΩ::AbstractArray)
    return (NO_FIELDS, DoesNotExist(), ΔΩ)
  end

  function ITensor_pullback(ΔΩ)
    return ITensor_pullback(ΔΩ.store)
  end

  ITensor_pullback(ΔΩ::Base.RefValue) = ITensor_pullback(ΔΩ[])

  return ITensor(st, is), ITensor_pullback
end

function rrule(::typeof(itensor), A::Array,
               i::Vararg{<:Any, N}) where {N}
  is = IndexSet(i...)

  function itensor_pullback(ΔΩ::ITensor)
    return (NO_FIELDS, array(ΔΩ),
            ntuple(_ -> DoesNotExist(), Val(N))...)
  end

  function itensor_pullback(ΔΩ::Union{<:Array, <:TensorStorage})
    return itensor_pullback(itensor(ΔΩ, is))
  end

  function itensor_pullback(ΔΩ::NamedTuple{(:store, :inds), Tuple{T, Nothing}} where {T})
    return itensor_pullback(ΔΩ.store)
  end

  itensor_pullback(ΔΩ::Base.RefValue) =
    itensor_pullback(ΔΩ[])

  return itensor(A, i...), itensor_pullback
end

#function rrule(::typeof(setinds), A::ITensor, is)
#  Ais = inds(A)
#
#  function setinds_pullback(ΔΩ::ITensor)
#    return (NO_FIELDS, setinds(ΔΩ, Ais), DoesNotExist())
#  end
#
#  setinds_pullback(ΔΩ::TensorStorage) =
#    setinds_pullback(itensor(ΔΩ, Ais))
#
#  setinds_pullback(ΔΩ::NamedTuple) =
#    setinds_pullback(ΔΩ.store)
#
#  setinds_pullback(ΔΩ::Base.RefValue) = setinds_pullback(ΔΩ[])
#
#  return setinds(A, is), setinds_pullback
#end

function _rrule_itensor(::typeof(*), A, B)

  if A isa Number
    indsΔΩ = inds(B)
  elseif B isa Number
    indsΔΩ = inds(A)
  else
    indsΔΩ = noncommoninds(A, B)
  end

  function times_pullback(ΔΩ)
    ∂A = ΔΩ * B
    ∂B = A * ΔΩ
    #@assert hassameinds(A, ∂A)
    #@assert hassameinds(B, ∂B)
    return (NO_FIELDS, ∂A, ∂B)
  end

  function times_pullback(ΔΩ::Base.RefValue)
    #return times_pullback(itensor(ΔΩ[].store))
    return times_pullback(itensor(ΔΩ[].store, indsΔΩ))
  end

  return A * B, times_pullback
end

rrule(::typeof(*), A, B::ITensor) =
  _rrule_itensor(*, A, B)

rrule(::typeof(*), A::ITensor, B) =
  _rrule_itensor(*, A, B)

rrule(::typeof(*), A::ITensor, B::ITensor) =
  _rrule_itensor(*, A, B)

function rrule(::typeof(+), A::ITensor, B::ITensor)
  function plus_pullback(ΔΩ)
    return (NO_FIELDS, ΔΩ, ΔΩ)
  end
  return A + B, plus_pullback
end

# XXX TODO: This is needed for conj(::NamedTuple) error
# Maybe instead define overload of conj(::NamedTuple)
function rrule(::typeof(dag), T::ITensor)
  function dag_pullback(ȳ::ITensor)
    return (NO_FIELDS, ȳ)
  end
  return dag(T), dag_pullback
end

# TODO: this is needed to overload the Zygote version
# of adjoint
@adjoint function Base.adjoint(T::ITensor)
  indsT = inds(T)

  function adjoint_pullback(ΔΩ::ITensor)
    return (setinds(ΔΩ, indsT),)
  end

  function adjoint_pullback(ΔΩ::NamedTuple{(:store, :inds), Tuple{StoreT, Nothing}} where {StoreT <: TensorStorage})
    return adjoint_pullback(itensor(ΔΩ.store, indsT))
  end

  adjoint_pullback(ΔΩ::Base.RefValue) = adjoint_pullback(ΔΩ[])

  return prime(T), adjoint_pullback
end

# TODO: for some reason this version isn't being used
# by Zygote
#function rrule(::typeof(adjoint), T::ITensor)
#  indsT = inds(T)
#
#  function adjoint_pullback(ΔΩ::ITensor)
#    return (NO_FIELDS, setinds(ΔΩ, indsT))
#  end
#
#  function adjoint_pullback(ΔΩ::NamedTuple{(:store, :inds), Tuple{StoreT, Nothing}} where {StoreT <: TensorStorage})
#    return adjoint_pullback(itensor(ΔΩ.store, indsT))
#  end
#
#  adjoint_pullback(ΔΩ::Base.RefValue) = adjoint_pullback(ΔΩ[])
#
#  return prime(T), adjoint_pullback
#end

