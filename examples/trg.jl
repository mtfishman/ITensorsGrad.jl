using ITensors
using ITensorsGrad
using LinearAlgebra
using Zygote

T(β) = [exp(β^2) sin(β)^3
        β^4 cos(β^5)]

# ITensor
A(β, i = Index(2, "i")) = itensor(T(β), i', dag(i))

β = 2.1

#Z(β) = T(β)[1, 1]
#
#@show Z(β)
#@show Z'(β)

function Z(β)
  Tᵦ = T(β)
  U, S, V = svd(Tᵦ)
  return (U * V')[1, 2]
end

@show Z(β)
@show Z'(β)

function Z(β)
  i = Index(2, "i")
  Aᵦ = A(β, i)
  U, S, V = svd(Aᵦ, i')
  u = commonind(U, S)
  v = commonind(V, S)
  return (U * δ(u, v) * V)[1, 2]
end

@show Z(β)
@show Z'(β)

