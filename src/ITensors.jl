
# To get around Zygote keyword argument issue
function dag(T::ITensor)
  TT = conj(tensor(T))
  return itensor(store(TT), dag(inds(T)))
end

+(A::NamedTuple{(:store, :inds), Tuple{StoreT, Nothing}} where {StoreT <: TensorStorage}, B::ITensor) = itensor(A.store, inds(B)) + B

(A::TensorStorage + B::ITensor) = itensor(A, inds(B)) + B

(A::ITensor + B::TensorStorage) = A + itensor(B, inds(A))

# TODO: are these needed?
# TODO: define this as (A::Base.RefValue + B::ITensor) = A[] + B
# and then define conversion of NamedTuple to ITensor
(A::Base.RefValue + B::ITensor) = A[] + B

# TODO: are these needed?
(A::ITensor + B::Base.RefValue) = B + A

# TODO: is this needed?
adjoint(A::Base.RefValue) = adjoint(A[])

adjoint(A::NamedTuple{(:store, :inds), Tuple{ITensorT, Nothing}} where {ITensorT <: ITensor}) = prime(A)

# XXX Probably the wrong definition
#prime(A::NamedTuple{(:store, :inds), Tuple{ITensorT, Nothing}} where {ITensorT <: ITensor}) = prime(A.store)
prime(A::NamedTuple{(:store, :inds), Tuple{ITensorT, Nothing}} where {ITensorT <: ITensor}) = A.store

adjoint(A::NamedTuple{(:store, :inds), Tuple{T, Nothing}} where {T <: TensorStorage}) = A.store

itensor(A::ITensor) = A

setinds(A::NamedTuple{(:store, :inds), Tuple{ITensorT, Nothing}} where {ITensorT <: ITensor}, is) = setinds(A.store, is)

