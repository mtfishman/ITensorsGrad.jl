using ITensorsGrad
using ITensors
using Test
using Zygote

@testset "ITensorsGrad.jl" begin

  mat(x) = [x*exp(x) x*exp(-x); x*exp(-x) x*exp(x)]
  function iten(x, i)
    return itensor(mat(x), i', i)
  end

  @testset "Simple gradient" begin

    function f(x, i)
      T = iten(x, i)
      return T[i'=>1, i=>1]^2 + T[i'=>2, i=>1]^2
    end

    function g(x)
      M = mat(x)
      return M[1, 1]^2 + M[2, 1]^2
    end

    i = Index(2)
    x = 2.5
    @test gradient(x -> f(x, i), x)[1] ≈ gradient(g, x)[1]

  end

end
