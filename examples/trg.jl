using ITensors
using ITensorsGrad
using LinearAlgebra
using Zygote

#T(β) = [exp( β) exp(-β)
#        exp(-β) exp( β)]

T(β) = [β^2 β^2
        β^2 β^2]

# ITensor
A(β, i = Index(2, "i")) = itensor(T(β), i', dag(i))

β = 0.1

Z(β) = A(β)[1, 1]
@show Z(β)
@show Z'(β)

function Z(β)
  Tᵦ = T(β)
  U, S, V = svd(Tᵦ)
  return (U * Diagonal(S) * V')[1, 1]
end
@show Z(β)
@show Z'(β)

function Z(β)
  i = Index(2, "i")
  Aᵦ = A(β, i)
  U, S, V = svd(Aᵦ, (i,))
  u = commonind(U, S)
  v = commonind(V, S)
  return (U * S * V)[1, 1]
end
@show Z(β)
@show Z'(β)

