using ITensors
using ITensorsGrad
using Zygote

include(joinpath(ITensors.examples_dir(), "src", "ctmrg_isotropic.jl"))
include(joinpath(ITensors.examples_dir(), "src", "2d_classical_ising.jl"))


function κ(β; χmax = 20, cutoff = 1e-8, nsteps = 20)
  d = 2
  s = Index(d, "Site")
  sₕ = addtags(s, "horiz")
  sᵥ = addtags(s, "vert")
  T = ising_mpo(sₕ, sᵥ, β)

  # Inital environment
  χ0 = 1
  l = Index(χ0, "Link")
  lₕ = addtags(l, "horiz")
  lᵥ = addtags(l, "vert")
  Cₗᵤ = itensor([1.0], lᵥ, lₕ)
  Aₗ = itensor([1.0, 0.0], lᵥ, lᵥ', sₕ)

  Cₗᵤ, Aₗ = ctmrg(T, Cₗᵤ, Aₗ; χmax = χmax,
                              cutoff = cutoff,
                              nsteps = nsteps)
  lᵥ = commonind(Cₗᵤ, Aₗ)
  lₕ = uniqueind(Cₗᵤ, Aₗ)
  Aᵤ = replaceinds(Aₗ, lᵥ => lₕ, lᵥ' => lₕ', sₕ => sᵥ)
  ACₗ = Aₗ * Cₗᵤ * dag(Cₗᵤ')
  ACTₗ = mapprime(ACₗ * dag(Aᵤ') * T * Aᵤ, 2 => 1, 1 => 0)
  return (ACTₗ * dag(ACₗ))[]
end

β = 1.1 * βc
κᵦ = κ(β)

κ_exact = exp(-β * ising_free_energy(β))
@show κᵦ, κ_exact
@show abs(κᵦ - κ_exact)

∂κᵦ = κ'(β)
@show hᵦ = -∂κᵦ / κᵦ

