# A bang of a def that returns Unit gives Unit{} and consumes its
# argument: a page frees the argument on the host, where it is, and does
# not ask the device. A bang after it still runs there.
import Base

def build(+d: Nat, +c: U32) -> Image:
  match d:
    case 0n:
      Pix{c}
    case 1n+p:
      Qua{build(p, c), build(p, U32.add(c, 1)), build(p, U32.add(c, 2)),
        build(p, U32.add(c, 3))}

def sum(img: Image) -> U32:
  match img:
    case Pix{c}:
      c
    case Qua{tl, tr, bl, br}:
      U32.add(U32.add(sum(tl), sum(tr)), U32.add(sum(bl), sum(br)))

def after(u: Unit) -> U32:
  match u:
    case Unit{}:
      sum(build!(4n, 1))

def main() -> U32:
  after(Image.drop!(build(6n, 0)))

#|1792
