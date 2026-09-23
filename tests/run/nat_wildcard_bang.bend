# A `case _` after `case 1n` binds a numbered name, which the C emitter
# spells _<n>_<k>: WGSL names cannot start with a digit, so the WebGPU
# lane keeps its underscore (app_ray_tracer_3d's shader failed to parse).
import Base

def pick(i: Nat, +x: U32) -> U32:
  match i:
    case 0n:
      x
    case 1n:
      (x + 1 : U32)
    case _:
      (x + 2 : U32)

def sum(+d: Nat, +n: U32) -> U32:
  match d:
    case 0n:
      pick(U32.to_nat((n % 3 : U32)), n)
    case 1n+p:
      a b = sum(p, (n * 2 : U32)) sum(p, (n * 2 + 1 : U32))
      (a + b : U32)

def main() -> U32:
  sum!(4n, 0)

#|135
