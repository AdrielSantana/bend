# A page keeps the Image a bang returns on the device when no code the
# host runs opens one: a later bang that takes it gets it back, a drop
# only uncounts it, and a bang while one is kept copies its own back.
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

def main() -> IO(Unit):
  do IO<Unit>:
    a : Image <- IO.pure(Image, build!(4n, 1))
    b : Image <- IO.pure(Image, build!(3n, 5))
    Unit <- IO.pure(Unit, Image.drop!(b))
    s : U32 <- IO.pure(U32, sum!(a))
    c : Image <- IO.pure(Image, build!(2n, 7))
    Unit <- IO.pure(Unit, Image.drop!(c))
    d : Image <- IO.pure(Image, build!(2n, 9))
    IO.print(U32.show(s) ++ " " ++ U32.show(sum!(d)))

#|1792 192
