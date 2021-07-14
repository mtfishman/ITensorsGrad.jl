using ITensorsGrad
using Documenter

DocMeta.setdocmeta!(
  ITensorsGrad, :DocTestSetup, :(using ITensorsGrad); recursive=true
)

makedocs(;
  modules=[ITensorsGrad],
  authors="Matthew Fishman <mfishman@flatironinstitute.org>",
  repo="https://github.com/mtfishman/ITensorsGrad.jl/blob/{commit}{path}#{line}",
  sitename="ITensorsGrad.jl",
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://mtfishman.github.io/ITensorsGrad.jl",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(;
  repo="github.com/mtfishman/ITensorsGrad.jl.git", devbranch="main", push_preview=true
)
