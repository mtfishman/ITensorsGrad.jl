
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

