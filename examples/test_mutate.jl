using Zygote

mutable struct MyVector{ElT}
  v::Vector{ElT}
end

function f1(β, N = 5)
  v = [β for n in 1:N]
  map!(vₙ -> vₙ^2, v, v)
  return sum(v)
end

function f2(β, N = 5)
  v = [β for n in 1:N]
  v = map(vₙ -> vₙ^2, v)
  return sum(v)
end

function f3(β, N = 5)
  v = MyVector([β for n in 1:N])
  map!(vₙ -> vₙ^2, v, v)
  return sum(v)
end

function f4(β, N = 5)
  v = MyVector([β for n in 1:N])
  v = map(vₙ -> vₙ^2, v)
  return sum(v)
end

