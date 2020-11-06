using FiniteDifferences # For testing gradients
using ITensors
using ITensorsGrad
using LinearAlgebra
using Test
using Zygote

@testset "Zygote ITensors" begin
  # Matrix
  T(β) = [β ^ 2 β ^ 2
          β ^ 2 β ^ 2]

  # ITensor
  function A(β, i = Index(2, "i"))
    return itensor(T(β), i', dag(i))
  end

  β = 3.5

  @show β

  ##############################################
  # BROKEN
  #

  # XXX this gives the wrong result, the version with ' works
  Z(β) = tr(T(β) * T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (mapprime(prime(Aᵦ, 2), 3 => 0) * prime(Aᵦ) * Aᵦ)[])
  @test Z(β) ≈ Zit(β)
  @test_broken Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); tr(Aᵦ' * Aᵦ; plev = 0 => 2))
  @test Z(β) ≈ Zit(β)
  @test_broken Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); tr(prime(Aᵦ) * Aᵦ; plev = 0 => 2))
  @test Z(β) ≈ Zit(β)
  @test_broken Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); tr(Aᵦ' * Aᵦ, 0 => 2))
  @test_broken Z(β) ≈ Zit(β)
  @test_broken Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); tr(prime(Aᵦ) * Aᵦ, 0 => 2))
  @test_broken Z(β) ≈ Zit(β)
  @test_broken Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); tr(product(Aᵦ, Aᵦ)))
  @test Z(β) ≈ Zit(β)
  @test_broken Z'(β) ≈ Zit'(β)






  ##############################################
  Z(β) = (T(β) * T(β))[1, 1]
  Zit(β) = (Aᵦ = A(β); (Aᵦ' * Aᵦ)[1, 1])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (T(β) * T(β))[1, 1]
  Zit(β) = (Aᵦ = A(β); (prime(Aᵦ) * Aᵦ)[1, 1])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (mapprime(Aᵦ' * Aᵦ, 2 => 1) * dag(δ(inds(Aᵦ))))[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (mapprime(prime(Aᵦ) * Aᵦ, 2 => 1) * dag(δ(inds(Aᵦ))))[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (Aᵦ' * Aᵦ * dag(δ(mapprime(inds(Aᵦ), 1 => 2))))[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (prime(Aᵦ) * Aᵦ * dag(δ(mapprime(inds(Aᵦ), 1 => 2))))[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (T(β) + 2 * T(β) * T(β))[1, 1]
  Zit(β) = (Aᵦ = A(β); (mapprime(Aᵦ, 1 => 2) + 2 * Aᵦ' * Aᵦ)[1, 1])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (2 * T(β) * T(β))[1, 1]
  Zit(β) = (Aᵦ = A(β); (2 * Aᵦ' * Aᵦ)[1, 1])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (2 * T(β) * T(β))[1, 1]
  Zit(β) = (Aᵦ = A(β); (2 * prime(Aᵦ) * Aᵦ)[1, 1])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (mapprime(Aᵦ'', 3 => 0) * Aᵦ' * Aᵦ)[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (mapprime(Aᵦ', 2 => 0) * Aᵦ)[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (mapprime(prime(Aᵦ), 2 => 0) * Aᵦ)[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = T(β)[1, 1]
  Zit(β) = swapprime(A(β), 0 => 1)[1, 1]
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (T(β) * T(β))[1, 1]
  Zit(β) = (i = Index(2); (A(β, i') * A(β, i))[1, 1])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = T(β)[1, 1]
  Zit(β) = A(β)[1, 1]
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = T(β)[1, 1]
  Zit(β) = A(β)'[1, 1]
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = T(β)[1, 1]^2 + T(β)[2, 1]^2
  Zit(β) = (Aᵦ = A(β); (i′, i) = inds(Aᵦ); Aᵦ[i′=>1, i=>1]^2 + Aᵦ[i′=>2, i=>1]^2)
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (3 * T(β) * T(β))[1, 1]
  Zit(β) = (i = Index(2, "i"); (3 * A(β, i') * A(β, i))[1, 1])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (i = Index(2, "i"); (dag(δ(i'', dag(i))) * A(β, i') * A(β, i))[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β) * T(β))
  Zit(β) = (i = Index(2, "i"); (dag(δ(i''', dag(i))) * A(β, i'') * A(β, i') * A(β, i))[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (T(β) * 3)[1, 1]
  Zit(β) = (A(β) * 3)[1, 1]
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (T(β) + 2 * T(β))[1, 1]
  Zit(β) = (Aᵦ = A(β); (Aᵦ + 2 * Aᵦ)[1, 1])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β))
  Zit(β) = (Aᵦ = A(β); (Aᵦ * dag(δ(inds(Aᵦ))))[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (Aᵦ * Aᵦ)[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(2 * T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (2 * Aᵦ * Aᵦ)[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (2 * T(β))[1, 1]
  Zit(β) = (2 * A(β))[1, 1]
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = (T(β) + 2 * T(β))[1, 1]
  Zit(β) = (Aᵦ = A(β); (Aᵦ + 2 * Aᵦ)[1, 1])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (Aᵦ * dag(Aᵦ))[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)

  Z(β) = tr(T(β) * T(β))
  Zit(β) = (Aᵦ = A(β); (swapprime(Aᵦ, 0 => 1) * Aᵦ)[])
  @test Z(β) ≈ Zit(β)
  @test Z'(β) ≈ Zit'(β)
end

