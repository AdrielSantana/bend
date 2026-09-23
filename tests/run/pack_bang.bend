# What a bang gives back, as a page packs it on the device: a node every
# lane of a round meets at once (a shared cell claimed by one, taken by the
# others a round later), a tower of shared cells, a long list (one lane
# walks it), a word array and an array of shared trees longer than a chunk
# (a lane a chunk) and a closure; each bang twice, so a pack starts where
# the last one left the device.
import Base

type Tree is Data:
  Leaf{v: U32}
  Node{l: Tree, r: Tree}

type Ls is Data:
  End{}
  Link{v: U32, next: Ls}

def spread(+d: Nat, +s: Tree) -> Tree:
  match d:
    case 0n:
      s
    case 1n+p:
      a b = spread(p, s) spread(p, s)
      Node{a, b}

def spread.of(+d: Nat, +v: U32) -> Tree:
  spread(d, Node{Leaf{v}, Leaf{0}})

def tower(+d: Nat, +t: Tree) -> Tree:
  match d:
    case 0n:
      t
    case 1n+p:
      tower(p, Node{t, t})

def sum(t: Tree) -> U32:
  match t:
    case Leaf{v}:
      v
    case Node{l, r}:
      (sum(l) + sum(r) : U32)

def range(+n: Nat, +i: U32, acc: Ls) -> Ls:
  match n:
    case 0n:
      acc
    case 1n+p:
      range(p, (i + 1 : U32), Link{i, acc})

def total(xs: Ls) -> U32:
  match xs:
    case End{}:
      0
    case Link{v, next}:
      (v + total(next) : U32)

def fill(+d: Nat, +v: U32) -> Array<U32>:
  Array.new(U32, d, v)

def at(r: Array<U32> & U32) -> U32:
  (a, v) = r
  v

def grove(+d: Nat, +v: U32) -> Array<Tree>:
  Array.new(Tree, d, Node{Leaf{v}, Leaf{1}})

def pick(r: Array<Tree> & Tree) -> Tree:
  (a, t) = r
  t

def adder(+k: U32) -> U32 -> U32:
  x => (x + k : U32)

def main() -> U32:
  +a = (sum(spread.of!(12n, 5)) + sum(spread.of!(11n, 6)) : U32)
  +b = (sum(tower!(16n, Leaf{3})) + sum(tower!(15n, Leaf{2})) : U32)
  +c = (total(range!(100000n, 1, End{})) + total(range!(1000n, 7, End{})) : U32)
  +d = (at(Array.get(U32, fill!(12n, 9), 4095)) + at(Array.get(U32, fill!(3n, 4), 2)) : U32)
  +e = (adder!(40)(2) + adder!(1)(1) : U32)
  +f = (sum(pick(Array.get(Tree, grove!(10n, 7), 1000))) + sum(pick(Array.get(Tree, grove!(9n, 8), 3))) : U32)
  (a + b + c + d + e + f : U32)

#|705884190
