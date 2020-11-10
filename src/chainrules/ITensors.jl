#
# NDTensors rrules
#

function rrule(::Type{<:Dense}, v::AbstractVector)

  function Dense_pullback(ΔΩ)
    return (NO_FIELDS, data(ΔΩ))
  end

  return Dense(v), Dense_pullback
end

function rrule(::Type{<:Combiner}, args::Vararg{<:Any, N}) where {N}
  function Combiner_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return Combiner(args...), Combiner_pullback
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

function rrule(::typeof(svd), T::ITensor, args::Vararg{<:Any, N}) where {N}
  indsT = inds(T)
  F = svd(T, args...)
  U, S, V = F.U, F.S, F.V
  @show U
  @show S
  @show V

  function svd_pullback(ΔΩ)
    @show typeof(ΔΩ)

    Uₜ, Sₜ, Vₜ = tensor.((U, S, V))
    ΔΩₜ = svd_back(Uₜ, Sₜ, Vₜ, ΔΩ.U, ΔΩ.S, ΔΩ.V)

    @show ΔΩₜ
    @show indsT

    ΔT = itensor(ΔΩₜ, indsT)

    @show ΔT

    return (ΔT, ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  #error("rrule(svd, ::ITensor)")

  return F, svd_pullback
end

function rrule(::Type{<:TagSet}, args::Vararg{<:Any, N}) where {N}
  function TagSet_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return TagSet(args...), TagSet_pullback
end

function rrule(::Type{<:Index}, args::Vararg{<:Any, N}) where {N}
  function Index_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return Index(args...), Index_pullback
end

function rrule(::Type{<:IndexSet}, args::Vararg{<:Any, N}) where {N}
  function IndexSet_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return IndexSet(args...), IndexSet_pullback
end

function rrule(::typeof(unioninds), args::Vararg{<:Any, N}) where {N}
  function unioninds_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return unioninds(args...), unioninds_pullback
end

function rrule(::typeof(commoninds), args::Vararg{<:Any, N}) where {N}
  function commoninds_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return commoninds(args...), commoninds_pullback
end

function rrule(::typeof(uniqueinds), args::Vararg{<:Any, N}) where {N}
  function uniqueinds_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return uniqueinds(args...), uniqueinds_pullback
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

# XXX don't seem to need this, uses ITensor constructor anyway
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
  indsT = inds(T)

  function dag_pullback(ΔΩ::ITensor)
    # XXX TODO: double check this definition
    return (NO_FIELDS, dag(ΔΩ))
  end

  function dag_pullback(ΔΩ::NamedTuple{(:store, :inds),
                                       Tuple{T, Nothing}} where {T <: TensorStorage})
    return dag_pullback(itensor(ΔΩ.store, indsT))
  end

  dag_pullback(ΔΩ::Base.RefValue) = dag_pullback(ΔΩ[])

  return dag(T), dag_pullback
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

