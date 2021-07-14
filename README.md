# ITensorsGrad

This extends the `ITensors.jl` library to make operations involving ITensors differentiable.

It uses `ChainRulesCore` to define reverse-mode AD rules for differentiating basic ITensor operations,
like contraction, addition, priming, tagging, and construction from arrays. Then, if an AD library
like Zygote.jl is used that recognizes rules written with the ChainRules interface, functions
making use of those ITensor operations that we currently support will be differentiable.

For example:
```julia
using ITensors
using ITensorsGrad
using Zygote

function f(x)
  i = Index(2, "i")
  T = itensor([exp(-x) exp(x); exp(x) exp(-x)], i', dag(i))
  Z = T'' * T' * T
  Z *= δ(dag(inds(Z))) 
  return (Z[])^(1//3)
end

x = 0.3
@show f(x)
@show f'(x)
```
returns:
```julia
f(x) = 2.0733047296286964
(f')(x) = 0.4388827959004289
```
We can compare against the result obtained from finite differencing:
```julia
using FiniteDifferences
function FiniteDifferences.to_vec(A::ITensor)
  function vec_to_ITensor(x)
    return isempty(inds(A)) ? ITensor(x[]) : itensor(x, inds(A))
  end
  return vec(array(A)), vec_to_ITensor
end

@show gradient(f, x)[1]
```
returns:
```julia
(gradient(f, x))[1] = 0.4388827959004289
```

## Limitations

For now, this package does not yet support tensor decompositions like SVD or QR. Stay tuned!

The same functionality is available in [ITensorNetworkAD.jl](https://github.com/ITensor/ITensorNetworkAD.jl), which will also provide more network-level optimizations of tensor contractions and derivatives through tensor contractions and for now will be more actively developed.

