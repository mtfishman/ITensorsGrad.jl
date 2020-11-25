using FiniteDifferences # For testing gradients
using ITensors
using ITensorsGrad
using LinearAlgebra
using Test
using Zygote

@testset "Zygote ITensors" begin
  T(β) = [exp(β) sin(β)
          β^2 cos(β)]
  A(β, i = Index(2, "i")) = itensor(T(β), i', dag(i))
  β = 1.2

  @testset "SVD" begin
    function Z(β)
      Tᵦ = T(β)
      U, S, V = svd(Tᵦ)
      return (U * V')[1, 2]
    end
    function Zit(β)
      i = Index(2, "i")
      Aᵦ = A(β, i)
      U, S, V = svd(Aᵦ, i')
      u = commonind(U, S)
      v = commonind(V, S)
      return (U * δ(u, v) * V)[1, 2]
    end
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    function Z(β)
      Tᵦ = T(β)
      U, S, V = svd(Tᵦ)
      return tr(U * Diagonal(S) * V')
    end
    function Zit(β)
      i = Index(2, "i")
      Aᵦ = A(β, i)
      U, S, V = svd(Aᵦ, i')
      return (U * S * V * δ(inds(Aᵦ)))[]
    end
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)
  end
end

