using ITensors
using ITensorsGrad
using LinearAlgebra
using Zygote

T(β) = [exp(β^2) sin(β)^3
        β^4 cos(β^5)]
A(β, i = Index(2, "i")) = itensor(T(β), i', dag(i))
β = 2.1

Z(β) = (T(β) + 2 * T(β))[1, 1]
Zit(β) = (Aᵦ = A(β); (Aᵦ + 2 * Aᵦ)[1, 1])
#Z(β) = (T(β) + T(β))[1, 1]
#Zit(β) = (Aᵦ = A(β); (Aᵦ + Aᵦ)[1, 1])
@show Z(β)
@show Zit(β)
@show Z'(β)
@show Zit'(β)

