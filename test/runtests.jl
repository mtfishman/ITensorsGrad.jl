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

  @testset "Contraction" begin
    ##############################################
    # BROKEN TESTS
    #

    # XXX this gives the wrong result
    Z(β) = tr(T(β) * T(β))
    function Zit(β)
      i = Index(2, "i")
      return (dag(δ(i'', dag(i))) * A(β, i') * A(β, i))[]
    end
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    # XXX this gives the wrong result
    Z(β) = tr(T(β) * T(β))
    Zit(β) = (Aᵦ = A(β); (mapprime(Aᵦ', 2 => 0) * Aᵦ)[])
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    # XXX this gives the wrong result
    Z(β) = tr(T(β) * T(β))
    Zit(β) = (Aᵦ = A(β); (mapprime(prime(Aᵦ), 2 => 0) * Aᵦ)[])
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    # XXX this gives the wrong result
    Z(β) = tr(T(β) * T(β))
    Zit(β) = (Aᵦ = A(β); (swapprime(Aᵦ, 0 => 1) * Aᵦ)[])
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    # XXX this gives the wrong result
    Z(β) = tr(T(β) * T(β) * T(β))
    function Zit(β)
      i = Index(2, "i")
      return (dag(δ(i''', dag(i))) * A(β, i'') * A(β, i') * A(β, i))[]
    end
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    # XXX this gives the wrong result
    Z(β) = tr(T(β) * T(β) * T(β))
    Zit(β) = (Aᵦ = A(β); (mapprime(Aᵦ'', 3 => 0) * Aᵦ' * Aᵦ)[])
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    # XXX this gives the wrong result, the version with ' works
    Z(β) = tr(T(β) * T(β) * T(β))
    Zit(β) = (Aᵦ = A(β); (mapprime(prime(Aᵦ, 2), 3 => 0) * prime(Aᵦ) * Aᵦ)[])
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)


    #
    # XXX: adjoint of tr(::ITensor) not working,
    # results in `nothing`. Maybe it is too complicated
    # for Zygote.
    #

    Z(β) = tr(T(β) * T(β))
    Zit(β) = (Aᵦ = A(β); tr(Aᵦ' * Aᵦ; plev = 0 => 2))
    @test Z(β) ≈ Zit(β)
    @test_broken Z'(β) ≈ Zit'(β)

    Z(β) = tr(T(β) * T(β))
    Zit(β) = (Aᵦ = A(β); tr(prime(Aᵦ) * Aᵦ; plev = 0 => 2))
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

    Z(β) = tr(T(β)' * T(β))
    Zit(β) = (Aᵦ = A(β); (Aᵦ * Aᵦ)[])
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    Z(β) = tr(2 * T(β)' * T(β))
    Zit(β) = (Aᵦ = A(β); (2 * Aᵦ * Aᵦ)[])
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    Z(β) = (2 * T(β))[1, 1]
    Zit(β) = (2 * A(β))[1, 1]
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)

    Z(β) = tr(T(β)' * T(β))
    Zit(β) = (Aᵦ = A(β); (Aᵦ * dag(Aᵦ))[])
    @test Z(β) ≈ Zit(β)
    @test Z'(β) ≈ Zit'(β)
  end

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

