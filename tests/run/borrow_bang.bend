# A bang's borrowed argument stays its caller's: a page copies it to the
# device and frees only the owned ones where they are. Freed there, the
# tree the next bang reads would be the one each churn builds over it.
# sum is only given trees held after it, so its def borrows; churn's goes
# to total, which owns it.
import Base

type Tree is Data:
  Leaf{v: U32}
  Node{l: Tree, r: Tree}

def build(+d: Nat, +c: U32) -> Tree:
  match d:
    case 0n:
      Leaf{c}
    case 1n+p:
      Node{build(p, (c * 2 : U32)), build(p, (c * 2 + 1 : U32))}

def sum(t: Tree) -> U32:
  match t:
    case Leaf{v}:
      v
    case Node{l, r}:
      a b = sum(l) sum(r)
      (a + b : U32)

def seen(+t: Tree) -> U32 & Tree:
  (sum!(t), t)

def total(t: Tree) -> U32:
  match t:
    case Leaf{v}:
      v
    case Node{l, r}:
      (total(l) + total(r) : U32)

def churn(+c: U32) -> U32:
  total(build(8n, c))

def fold(+acc: U32, r: U32 & Tree) -> U32 & Tree:
  (s, t) = r
  ((acc + s + churn(acc) : U32), t)

def round(r: U32 & Tree) -> U32 & Tree:
  (acc, t) = r
  fold(acc, seen(t))

def last(r: U32 & Tree) -> U32:
  (acc, t) = r
  (acc + sum(t) : U32)

def main() -> U32:
  last(round(round(round((0, build(6n, 1))))))

#|3324108288
