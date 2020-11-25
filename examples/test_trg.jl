using ITensors
using ITensorsGrad
using Zygote

include(joinpath(ITensors.examples_dir(),
                 "src", "2d_classical_ising.jl"))

function trg_one_step(T;
                      χmax = χmax,
                      nsteps = nsteps,
                      cutoff = 1e-15)
  sₕ, sᵥ = filterinds(T; plev = 0)

  Fₕ, Fₕ′ = factorize(T, (sₕ', sᵥ');
                      ortho = "none",
                      maxdim = χmax,
                      cutoff = cutoff,
                      tags = tags(sₕ))

  s̃ₕ = commonind(Fₕ, Fₕ′)
  Fₕ′ *= δ(dag(s̃ₕ), s̃ₕ')

  Fᵥ, Fᵥ′ = factorize(T, (sₕ, sᵥ');
                      ortho = "none",
                      maxdim = χmax,
                      cutoff = cutoff,
                      tags = tags(sᵥ))

  s̃ᵥ = commonind(Fᵥ, Fᵥ′)
  Fᵥ′ *=  δ(dag(s̃ᵥ), s̃ᵥ')

  T = (Fₕ * δ(dag(sₕ'), sₕ)) *
      (Fᵥ * δ(dag(sᵥ'), sᵥ)) *
      (Fₕ′ * δ(dag(sₕ), sₕ')) *
      (Fᵥ′ * δ(dag(sᵥ), sᵥ'))

  return (T * δ(s̃ₕ, s̃ₕ') * δ(s̃ᵥ, s̃ᵥ'))[]
end

function κ(β; χmax = 2, nsteps = 2)
  d = 2
  s = Index(d)
  sₕ = addtags(s, "horiz")
  sᵥ = addtags(s, "vert")
  Tᵦ = ising_mpo(sₕ, sᵥ, β)
  κᵦ = trg_one_step(Tᵦ; χmax = χmax, nsteps = nsteps)
  return κᵦ
end

β = 1.1 * βc

κᵦ = κ(β)

κ_exact = exp(-β * ising_free_energy(β))
@show κᵦ, κ_exact
@show abs(κᵦ - κ_exact)

∂κᵦ = @show κ'(β)
hᵦ = @show -∂κᵦ / κᵦ

