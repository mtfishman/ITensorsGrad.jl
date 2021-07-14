# ITensorsGrad

This extends the `ITensors.jl` library to make operations involving ITensors differentiable.

It uses `ChainRulesCore` to define reverse-mode AD rules for differentiating basic ITensor operations,
like contraction, addition, priming, tagging, and construction from arrays. Then, if an AD library
like Zygote.jl is used that recognizes rules written with the ChainRules interface, functions
making use of those ITensor operations that we currently support will be differentiable.

For example:
```julia
```

