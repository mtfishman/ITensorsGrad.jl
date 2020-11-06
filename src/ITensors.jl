
# To get around Zygote keyword argument issue
function dag(T::ITensor)
  TT = conj(tensor(T))
  return itensor(store(TT), dag(inds(T)))
end

# TODO: are these needed?
# TODO: define this as (A::Base.RefValue + B::ITensor) = A[] + B
# and then define conversion of NamedTuple to ITensor
(A::Base.RefValue + B::ITensor) =
  itensor(A[].store) + B

# TODO: are these needed?
(A::ITensor + B::Base.RefValue) = B + A

# TODO: is this needed?
adjoint(A::Base.RefValue) = adjoint(A[])

adjoint(A::NamedTuple{(:store, :inds), Tuple{ITensorT, Nothing}} where {ITensorT <: ITensor}) = prime(A)

prime(A::NamedTuple{(:store, :inds), Tuple{ITensorT, Nothing}} where {ITensorT <: ITensor}) = prime(A.store)

adjoint(A::NamedTuple{(:store, :inds), Tuple{T, Nothing}} where {T <: TensorStorage}) = A.store

itensor(A::ITensor) = A

