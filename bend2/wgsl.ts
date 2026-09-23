// Wgsl
// ====

// A page's `!` on WebGPU. WebGPU has no 64-bit integer, no fence between
// workgroups, no forward progress and no heap shared with the host
// (bendlang/bend#920), so this lane is not a port of bend_dev. The device C
// of the CUDA dialect goes through clang's typed AST into WGSL: a u64 is a
// vec2<u32>, the corpus one array<atomic<u32>>, a pointer an index into it.
// An atomic load skips the cache and made a round four times slower, so
// every store also goes to a plain mirror, P, where the u64 loads read.
// A round is one dispatch that runs each queued task on a lane and queues
// what it makes (a fork's kids, a parent its last kid joined) for the next,
// so no lane reads what another wrote in the same dispatch. A bang copies
// its arguments into a device arena, and its result back into the host's
// heap (GLUE).

import * as child from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

// Types
// =====

// A clang JSON AST node.
type N = { kind: string; inner?: N[]; [k: string]: any };

type Ty =
  | { k: "void" }
  | { k: "bool" }
  | { k: "int"; w: number; s: boolean }
  | { k: "f32" }
  | { k: "ptr"; to: Ty }
  | { k: "arr"; of: Ty; n: number }
  | { k: "rec"; name: string; union: boolean; fs: Fld[]; size: number;
      al: number };

type Fld = { name: string; t: Ty; off: number };

// Where a pointer points: into the corpus (a u32 index), at a WGSL
// variable or element (its reference), into a local array (its reference
// and an element offset), or into a table of TAB (its first word).
type Ptr =
  | { pk: "heap"; ix: string }
  | { pk: "ref"; r: string }
  | { pk: "arr"; r: string; off: string; n: number }
  | { pk: "tab"; at: number };

// A value: WGSL text in its type's representation, or a pointer. `b` marks
// a C int held as a WGSL bool (a comparison, a logical operator).
type Val = { s: string; t: Ty; b?: boolean; p?: Ptr };

// A place: a WGSL reference (a variable, a param's (*p), an element), the
// corpus at a u32 index, a table's word, or a union's member over its bits.
type Place =
  | { lk: "ref"; r: string; t: Ty }
  | { lk: "heap"; ix: string; t: Ty }
  | { lk: "tab"; at: string; t: Ty }
  | { lk: "bits"; r: string; t: Ty };

// A local: its WGSL name and C type.
type Local = { name: string; t: Ty };

type Unit = {
  tds: Map<string, N>;
  recs: Map<string, N>;
  tags: Map<string, string>;
  locs: Map<string, string>;
  fns: Map<string, N>;
  globs: Map<string, N>;
  tys: Map<string, Ty>;
  tabs: Map<string, number>;
  tab: number[];
  done: Map<string, string>;
  todo: [string, N, Ptr[]][];
  pure: Map<string, boolean>;
};

type Fn = {
  u: Unit;
  out: string[];
  ind: string;
  locals: Map<string, Local>;
  names: Map<string, number>;
  tmp: number;
  ret: Ty;
  name: string;
  alias: Map<string, Ptr>;
};

// Constants
// =========

const VOID: Ty = { k: "void" };
const BOOL: Ty = { k: "bool" };
const F32: Ty = { k: "f32" };
const U32: Ty = { k: "int", w: 32, s: false };
const I32: Ty = { k: "int", w: 32, s: true };
const U64: Ty = { k: "int", w: 64, s: false };

const SCALARS: Record<string, Ty> = Object.setPrototypeOf({
  "void": VOID, "_Bool": BOOL, "bool": BOOL, "float": F32,
  "char": { k: "int", w: 8, s: true },
  "signed char": { k: "int", w: 8, s: true },
  "unsigned char": { k: "int", w: 8, s: false },
  "short": { k: "int", w: 16, s: true },
  "unsigned short": { k: "int", w: 16, s: false },
  "int": I32, "unsigned int": U32,
  "long": { k: "int", w: 64, s: true }, "unsigned long": U64,
  "long long": { k: "int", w: 64, s: true }, "unsigned long long": U64,
}, null);

// The CUDA builtins the dialect calls, declared so clang types them; the
// math takes and gives f32, as CUDA's overloads do.
const MATH1 = ["sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan",
  "asin", "acos", "atan", "sinh", "cosh", "tanh", "floor", "ceil", "trunc",
  "fabs"];
const MATH2 = ["pow", "atan2", "fmod"];
const ATOMS = ["Add", "Sub", "And", "Or", "Xor", "Min", "Max"];

const PRELUDE = [
  ...ATOMS.map((a) => `unsigned atomic${a}(unsigned*, unsigned);`),
  "unsigned atomicCAS(unsigned*, unsigned, unsigned);",
  "void __threadfence(void);",
  "void __syncthreads(void);",
  "int __clz(int);",
  ...MATH1.map((m) => `float ${m}(float);`),
  ...MATH2.map((m) => `float ${m}(float, float);`),
].join("\n") + "\n";

// The rounds' words in the header's first line, which the host's runtime
// leaves free, and their queues, in the rings' region, which it leaves
// unused here.
const WG_DEFS = `
#define WG_SIDE  2
#define WG_NIN   3
#define WG_NOUT  4
#define WG_SEQ   5
#define WG_GRAB  6
#define WG_STOP  7
#define WG_ROUND 8
#define WG_ORD   9
#define WG_PACK  10
#define WG_PR0   11
#define WG_PTOP  12
#define WG_KEEP  13
#define WG_WIN   14
#define WG_PEND  0xFFFFFFFFu
#define WG_FWD   0x80000000u
#define WG_CHUNK 256u
#define WG_RING  16u
#define WG_BUDGET 64u
#define WG_BLK   (1ull << 63)
#define WG_LOCS  0x7FFFFFFFull
#define WG_QCAP  (65535u * 64u)
#define wg_qat(s)  (RING_OFF + (u64)(s) * WG_QCAP)
#define wg_q(H, s) ((H) + wg_qat(s))
#if defined(CID_QUA) && !HOST_IMAGE
#define WG_IMAGE CID_QUA
#else
#define WG_IMAGE 0x10000
#endif
`;

// The rounds, in the runtime's own C: a plan flips the queues and sizes
// the next dispatch, a lane a task up to the lanes, 0 when the root is
// done, an error was posted or nothing is queued. A lane runs its task, as
// monk_step does, then takes the next one nobody took, and queues what
// each leaves: a ready parent, a fork's kids. Tasks fork until there are
// four for each lane, then run whole, so the lanes finish together. Every
// handoff crosses a dispatch. The order a SIMD group runs its tasks in is
// worth 2 to 3 times, and which one wins is the program's: with WG_ORD a
// fork's kids take adjacent slots, so a group runs neighbours (Bendcraft's
// rays read the same blocks); without it a slot a kid, so a group runs the
// same kid of neighbouring parents (raytrace's hashed columns, alike in
// every row, cost alike).
//
// A lane that ran out of heap runs on over nodes that alias until it sees
// the error, and its stray writes can land in the header: no loop here
// trusts a count past the queue's room or runs on past an error: a
// dispatch that does not end can hold the GPU and the screen with it (a
// Mac froze, 2026-09-23, while a page's heap ran out bang after bang).
//
// Once the root is done, the rounds pack: they copy what the root's words
// reach into R, from the first page the rounds left unused, so the host
// reads R alone. A job is a word to copy and where to (src << 32 | dst). A
// lane walks a job's nodes depth first, as many as a budget, and queues
// the rest, so a tree spreads a round at a time and a job's nodes lie
// together in R. A shared cell is copied by the lane that claims its high
// word (WG_PEND), which then holds the copy (WG_FWD | loc): a reference
// that finds it claimed takes the copy a round later. A bang's cells count
// what the result holds of them, so a copy keeps its count. A block is
// copied a chunk a lane: a chunk's job is WG_BLK, a bit for words alone,
// its source and its destination. When the host keeps Images on the
// device (WG_KEEP), the pack leaves an Image where the rounds built it,
// for the host to draw it there.
const ROUNDS = WG_DEFS + `
static void wg_push(Corpus H, Term t, u32 at) {
  if (at >= WG_QCAP) {
    err_post(H, ERR_RING);
    return;
  }
  wg_q(H, a32_load(a32_at(H, WG_SIDE)) ^ 1)[at] = t;
}

static bool wg_kept(Corpus H, Term t) {
  return a32_load(a32_at(H, WG_KEEP)) != 0 && term_tag(t) == TAG_CTR
    && !term_rfc(t) && term_aux(t) == WG_IMAGE;
}

static Loc wg_room(Corpus H, u32 words, Loc r0, Loc end) {
  Loc r = r0 + a32_add(a32_at(H, WG_PTOP), words);
  if (r + words > end) {
    err_post(H, ERR_HEAP);
    return 0;
  }
  return r;
}

// A job's nodes, depth first from the one at s for d, WG_BUDGET at most,
// with a ring of WG_RING fields to visit: counted (base 0), or copied into
// R from base on. What the walk does not take is queued as jobs when it
// copies: a cell, a block longer than a chunk, the ring's oldest field when
// it is full, and the ring's fields at the budget. Both passes take the
// same path, so the count is the copy's words, and a job's nodes lie
// together in R, in the order the host walks them.
static u32 wg_walk(Corpus H, Loc s0, Loc d0, Loc base) {
  u64 ring[WG_RING];
  u32 lo   = 0;
  u32 hi   = 1;
  u32 used = 0;
  ring[0] = (s0 << 32) | d0;
  for (u32 b = 0; b < WG_BUDGET && hi != lo; b += 1) {
    hi -= 1;
    Loc  s   = ring[hi & (WG_RING - 1)] >> 32;
    Loc  d   = (u32)ring[hi & (WG_RING - 1)];
    Term t   = H[s];
    u64  tag = term_tag(t);
    u32  aux = (u32)term_aux(t);
    Loc  loc = term_loc(t);
    bool blk = tag == TAG_ARR || tag == TAG_BUF;
    if (!blk && tag != TAG_CTR && tag != TAG_CLO) {
      err_post(H, ERR_TAGS);
      return used;
    }
    u32 n  = tag == TAG_CTR ? cid_arity(aux) : tag == TAG_CLO
      ? fid_arity(aux) - 1 : 1u << blk_span(t);
    Loc nd = base + used;
    used += 1u << (blk ? blk_span(t) : cls_fit(n));
    if (base != 0) {
      H[d] = (t & ~LOC_MASK) | nd;
    }
    for (u32 j = 0; j < n; j += 1) {
      Term f  = H[loc + j];
      u64  ft = term_tag(f);
      u64  e  = ((loc + j) << 32) | (nd + j);
      if (tag == TAG_BUF || term_triv(f) || wg_kept(H, f)) {
        if (base != 0) {
          H[nd + j] = f;
        }
        continue;
      }
      if (term_rfc(f) || ((ft == TAG_ARR || ft == TAG_BUF)
        && (1u << blk_span(f)) > WG_CHUNK)) {
        if (base != 0) {
          wg_push(H, e, a32_add(a32_at(H, WG_NOUT), 1));
        }
        continue;
      }
      if (hi - lo == WG_RING) {
        if (base != 0) {
          wg_push(H, ring[lo & (WG_RING - 1)], a32_add(a32_at(H, WG_NOUT), 1));
        }
        lo += 1;
      }
      ring[hi & (WG_RING - 1)] = e;
      hi += 1;
    }
  }
  if (base != 0 && hi != lo) {
    u32 at = a32_add(a32_at(H, WG_NOUT), hi - lo);
    for (u32 i = lo; i != hi; i += 1) {
      wg_push(H, ring[i & (WG_RING - 1)], at + i - lo);
    }
  }
  return used;
}

// The term at s for d: a word as it is, a block longer than a chunk a
// chunk a job, else a walk counted, then copied into the room it takes.
static void wg_node(Corpus H, Loc s, Loc d, Loc r0, Loc end) {
  Term t   = H[s];
  u64  tag = term_tag(t);
  Loc  loc = term_loc(t);
  if (term_triv(t) || wg_kept(H, t)) {
    H[d] = t;
    return;
  }
  if ((tag == TAG_ARR || tag == TAG_BUF) && (1u << blk_span(t)) > WG_CHUNK) {
    u32 n  = 1u << blk_span(t);
    Loc nd = wg_room(H, n, r0, end);
    u32 at = a32_add(a32_at(H, WG_NOUT), n / WG_CHUNK);
    u64 bf = (u64)(tag == TAG_BUF) << 62;
    H[d] = (t & ~LOC_MASK) | nd;
    for (u32 c = 0; c < n && nd != 0; c += WG_CHUNK) {
      wg_push(H, WG_BLK | bf | ((loc + c) << 31) | (nd + c), at);
      at += 1;
    }
    return;
  }
  Loc base = wg_room(H, wg_walk(H, s, d, 0), r0, end);
  if (base != 0) {
    wg_walk(H, s, d, base);
  }
}

// A chunk of a block: its words as they are, its terms each a job.
static void wg_chunk(Corpus H, bool buf, Loc loc, Loc nd) {
  for (u32 j = 0; j < WG_CHUNK; j += 1) {
    Term f = H[loc + j];
    H[nd + j] = f;
    if (!buf && !term_triv(f)) {
      wg_push(H, ((loc + j) << 32) | (nd + j), a32_add(a32_at(H, WG_NOUT), 1));
    }
  }
}

// A cell's reference at s, for d. A max of WG_PEND claims the cell's high
// word, and the lane that finds it unclaimed copies the cell: the copy's
// word holds the content's term while it is walked, then the cell's word,
// the count kept. A reference that finds the copy takes it (and puts it
// back if its max hid it), and one that finds the cell claimed tries again
// a round later. A room at 0 is the heap run out, and word 0 the header's.
static void wg_cell(Corpus H, Loc s, Loc d, Loc r0, Loc end) {
  Term     t  = H[s];
  Loc      r  = term_loc(t);
  DEV u32* hi = a32_at(H, r) + 1;
  u32      v  = a32_load(hi);
  u32      o  = v < WG_FWD ? a32_max(hi, WG_PEND) : v;
  if (o != v && o != WG_PEND) {
    a32_store(hi, o);
  }
  if (o == WG_PEND) {
    wg_push(H, (s << 32) | d, a32_add(a32_at(H, WG_NOUT), 1));
    return;
  }
  if (o >= WG_FWD) {
    H[d] = (t & ~LOC_MASK) | (o & ~WG_FWD);
    return;
  }
  Loc c = wg_room(H, 1, r0, end);
  if (c == 0) {
    return;
  }
  u32  lo = a32_load(a32_at(H, r));
  Term ct = (t & ~(RFC_BIT | LOC_MASK)) | ((Loc)v << 8) | (lo >> 24);
  a32_store(hi, WG_FWD | (u32)c);
  H[d] = (t & ~LOC_MASK) | c;
  H[c] = ct;
  wg_node(H, c, c, r0, end);
  H[c] = (term_loc(H[c]) << 24) | (lo & RFC_CNT);
}

static void wg_pack(Corpus H, u64 job, Loc r0, Loc end) {
  Loc s = job >> 32;
  Loc d = (u32)job;
  if (job >> 63 != 0) {
    wg_chunk(H, (job >> 62 & 1) != 0, job >> 31 & WG_LOCS, job & WG_LOCS);
  } else if (term_rfc(H[s])) {
    wg_cell(H, s, d, r0, end);
  } else {
    wg_node(H, s, d, r0, end);
  }
}

// The root's words as the pack's first jobs, each copied onto itself.
static void wg_roots(Corpus H) {
  u32 n = 0;
  a32_store(a32_at(H, WG_PACK), 1);
  H[WG_PR0] = HEAP_OFF + ((Loc)a32_load(a32_at(H, H_BUMP)) << PAGE_BITS);
  for (u32 j = 0; j < WL_RESW && j + 1 < a32_load(a32_at(H, H_ROOT_DONE)); j += 1) {
    Loc w = H_ROOT_WORD + j;
    if (!term_triv(H[w])) {
      wg_push(H, (w << 32) | w, n);
      n += 1;
    }
  }
  a32_store(a32_at(H, WG_NOUT), n);
}

static u32 wg_plan(Corpus H) {
  if (a32_load(a32_at(H, WG_STOP)) != 0) {
    return 0;
  }
  if (root_done(H) && a32_load(a32_at(H, WG_PACK)) == 0) {
    wg_roots(H);
  }
  u32 n = a32_load(a32_at(H, WG_NOUT));
  u32 m = n < LANES ? n : LANES;
  if (n > WG_QCAP) {
    err_post(H, ERR_RING);
  }
  if (n == 0 || err_seen(H)) {
    a32_store(a32_at(H, WG_STOP), 1);
    return 0;
  }
  a32_add(a32_at(H, WG_ROUND), 1);
  a32_store(a32_at(H, WG_SEQ), n >= 4 * LANES);
  a32_store(a32_at(H, WG_NOUT), 0);
  a32_store(a32_at(H, WG_NIN), n);
  a32_store(a32_at(H, WG_GRAB), m);
  a32_store(a32_at(H, WG_SIDE), a32_load(a32_at(H, WG_SIDE)) ^ 1);
  return (m + 63) / 64;
}

static void wg_task(Corpus H, u32 i, Term t, u32 seq) {
  Env  e = { H, H + ALC_OFF + i };
  Term r = work_loop(e, (Stk)(H + STAK_OFF + i), t, seq);
  if (r == 0) {
    return;
  }
  Loc loc = term_loc(r);
  u32 ar  = fid_arity((u32)term_aux(r));
  if (a32_load(a32_at(H, loc + ar + 1)) == 0) {
    wg_push(H, r, a32_add(a32_at(H, WG_NOUT), 1));
    return;
  }
  bool adj = a32_load(a32_at(H, WG_ORD)) != 0;
  u32  at  = 0;
  if (adj) {
    u32 n = 0;
    for (u32 j = 0; j < ar; j += 1) {
      n += term_tag(H[loc + j]) == TAG_TSK;
    }
    at = a32_add(a32_at(H, WG_NOUT), n);
  }
  for (u32 j = 0; j < ar; j += 1) {
    Term k = H[loc + j];
    if (term_tag(k) == TAG_TSK) {
      if (!adj) {
        at = a32_add(a32_at(H, WG_NOUT), 1);
      }
      H[loc + j] = TERM_HOLE;
      wg_push(H, k, at);
      at += 1;
    }
  }
}

static void wg_run(Corpus H, u32 i) {
  u32 n    = a32_load(a32_at(H, WG_NIN));
  u32 seq  = a32_load(a32_at(H, WG_SEQ));
  u32 side = a32_load(a32_at(H, WG_SIDE));
  for (u32 j = i; j < n && !err_seen(H); j = a32_add(a32_at(H, WG_GRAB), 1)) {
    wg_task(H, i, wg_q(H, side)[j], seq);
  }
}

// The pack's rounds run in a kernel of their own: in the tasks' kernel the
// pack's code made Slash Boss 3D's first bang allocate past the heap in its
// eighth round, before any packing, on most runs.
static void wg_packs(Corpus H, u32 i) {
  u32 n    = a32_load(a32_at(H, WG_NIN));
  u32 side = a32_load(a32_at(H, WG_SIDE));
  Loc r0   = H[WG_PR0];
  Loc end  = HEAP_OFF + ((Loc)a32_load(a32_at(H, H_CAP)) << PAGE_BITS);
  for (u32 j = i; j < n && !err_seen(H); j = a32_add(a32_at(H, WG_GRAB), 1)) {
    wg_pack(H, wg_q(H, side)[j], r0, end);
  }
}

static u32 wg_packing(Corpus H) {
  return a32_load(a32_at(H, WG_PACK));
}

// A kept Image's square, a lane a pixel, as the native window_dev draws
// it: WG_WIN holds its node, then the band's width and rows, the square's
// level and the band's first row; the pixels go to the third queue, a row
// every 64 words, as a copy into a texture takes them.
static void wg_window(Corpus H, u32 i) {
  u32 wh = a32_load(a32_at(H, WG_WIN + 1));
  u32 kr = a32_load(a32_at(H, WG_WIN + 1) + 1);
  u32 w  = wh & 0xFFFF;
  if (i < w * (wh >> 16)) {
    ((DEV u32*)wg_q(H, 2))[i / w * ((w + 63) & ~63u) + i % w] = window_pix(H,
      H[WG_WIN], kr & 0xFF, i % w, (kr >> 8) + i / w);
  }
}
`;

// Die
// ===

function die(m: string): never {
  throw new Error("WGSL lane: " + m);
}

// Ast
// ===

// The device C: the template's CUDA dialect with ROUNDS appended,
// preprocessed, its C++ kernels cut (the rounds replace them), and typed by
// clang.
function ast_of(c: string, dir: string, cc: string): N {
  const src = path.join(dir, "dev.c");
  const pre = path.join(dir, "dev.i");
  const dev = path.join(dir, "dev2.c");
  fs.writeFileSync(src, c + ROUNDS);
  const run = (args: string[]): string => {
    const r = child.spawnSync(cc, args, { encoding: "utf8",
      maxBuffer: 2 ** 31 });
    if (r.status !== 0) {
      die(cc + " failed on the device C:\n" + (r.stderr ?? "").slice(0, 4000));
    }
    return r.stdout;
  };
  run(["-E", "-P", "-D__CUDACC_RTC__", "-DCUBE_LOG=7", "-x", "c", src, "-o",
    pre]);
  const ls = fs.readFileSync(pre, "utf8").split("\n");
  const keep: string[] = [];
  for (let i = 0; i < ls.length; i += 1) {
    if (!ls[i].startsWith("extern \"C\"")) {
      keep.push(ls[i]);
      continue;
    }
    let depth = 0;
    let open = false;
    for (; i < ls.length; i += 1) {
      depth += ls[i].split("{").length - ls[i].split("}").length;
      open ||= ls[i].includes("{");
      if (open && depth === 0) {
        break;
      }
    }
  }
  fs.writeFileSync(dev, PRELUDE + keep.join("\n"));
  return JSON.parse(run(["-x", "c", "-std=c2x", "-fno-builtin",
    "-fsyntax-only", "-Wno-everything", "-Xclang", "-ast-dump=json", dev]));
}

// Unit
// ====

function unit_new(ast: N): Unit {
  const u: Unit = { tds: new Map(), recs: new Map(), tags: new Map(),
    locs: new Map(),
    fns: new Map(), globs: new Map(), tys: new Map(), tabs: new Map(),
    tab: [], done: new Map(), todo: [], pure: new Map() };
  const walk = (n: N): void => {
    const xs = n.inner ?? [];
    if (n.kind === "DeclStmt" || n.kind === "TranslationUnitDecl") {
      let rec: string | null = null;
      for (const x of xs) {
        if (x.kind === "RecordDecl") {
          rec = x.id;
          u.recs.set(x.id, x);
        } else if (x.kind === "VarDecl" && rec !== null) {
          for (const s of [x.type.qualType, x.type.desugaredQualType]) {
            if (s?.includes("(unnamed")) {
              u.locs.set(s, rec);
            }
          }
        }
      }
    }
    xs.forEach(walk);
  };
  walk(ast);
  for (const d of ast.inner ?? []) {
    if (d.kind === "TypedefDecl") {
      u.tds.set(d.name, d);
      const id = d.inner?.[0]?.ownedTagDecl?.id;
      if (id !== undefined && !u.tags.has(id)) {
        u.tags.set(id, d.name);
      }
    } else if (d.kind === "FunctionDecl" && d.inner?.some((x) =>
      x.kind === "CompoundStmt")) {
      u.fns.set(d.name, d);
    } else if (d.kind === "VarDecl") {
      u.globs.set(d.name, d);
    }
  }
  return u;
}

// Ty
// ==

function ty_of(u: Unit, q: { qualType: string; desugaredQualType?: string })
  : Ty {
  return ty_str(u, q.desugaredQualType ?? q.qualType);
}

function ty_str(u: Unit, s: string): Ty {
  const hit = u.tys.get(s);
  if (hit !== undefined) {
    return hit;
  }
  const q = s.replace(/\b(const|volatile|restrict)\b/g, " ")
    .replace(/\s+/g, " ").trim();
  const rec = u.locs.get(s) ?? u.locs.get(q);
  const m = /^(.*?) ?\[(\d+)\]$/.exec(q);
  const td = q.replace(/^(struct|union) /, "");
  const t: Ty = rec !== undefined ? ty_rec(u, rec)
    : q.endsWith("*") ? { k: "ptr", to: ty_str(u, q.slice(0, -1)) }
    : m !== null ? { k: "arr", of: ty_str(u, m[1]), n: Number(m[2]) }
    : SCALARS[q] ?? (u.tds.has(td) ? ty_td(u, u.tds.get(td)!)
      : die("a C type it does not know: " + s));
  u.tys.set(s, t);
  return t;
}

function ty_td(u: Unit, d: N): Ty {
  let id: string | undefined;
  const find = (n: N): void => {
    id ??= n.ownedTagDecl?.id ?? (n.kind === "RecordType" ? n.decl?.id
      : undefined);
    (n.inner ?? []).forEach(find);
  };
  find(d);
  return id !== undefined && u.recs.has(id) ? ty_rec(u, id)
    : ty_str(u, d.type.desugaredQualType ?? d.type.qualType);
}

// A record laid out as C lays it: a field at its alignment, the size
// rounded to the widest; named by its typedef where it has one, so the
// kernels can build an Env.
function ty_rec(u: Unit, id: string): Ty {
  const d = u.recs.get(id) ?? die("a record it cannot find: " + id);
  const union = d.tagUsed === "union";
  const fs: Fld[] = [];
  let size = 0;
  let al = 1;
  for (const f of d.inner ?? []) {
    if (f.kind !== "FieldDecl") {
      continue;
    }
    const t = ty_of(u, f.type);
    const [fz, fa] = ty_size(t);
    const off = union ? 0 : Math.ceil(size / fa) * fa;
    fs.push({ name: f.name, t, off });
    size = union ? Math.max(size, fz) : off + fz;
    al = Math.max(al, fa);
  }
  const tag = u.tags.get(id);
  return { k: "rec", name: tag !== undefined ? "S_" + tag : "R"
    + id.replace(/^0x/, ""), union, fs, size: Math.ceil(size / al) * al, al };
}

function ty_size(t: Ty): [number, number] {
  switch (t.k) {
    case "int":
      return [t.w / 8, t.w / 8];
    case "f32":
      return [4, 4];
    case "ptr":
      return [8, 8];
    case "arr": {
      const [z, a] = ty_size(t.of);
      return [z * t.n, a];
    }
    case "rec":
      return [t.size, t.al];
    default:
      return [1, 1];
  }
}

// Its size in the corpus's u32 words: what the heap holds is whole words.
function ty_words(t: Ty): number {
  const z = ty_size(t)[0];
  return z % 4 === 0 ? z / 4 : die("a corpus access under a word wide");
}

function ty_wide(t: Ty): boolean {
  return t.k === "int" && t.w === 64;
}

// Its WGSL type: an int of 32 bits or less is a u32 or i32 holding a value
// of its range, a 64-bit one a vec2<u32> (low, high), a union its bits, a
// pointer into the corpus a u32 index.
function ty_wgsl(t: Ty): string {
  switch (t.k) {
    case "bool":
      return "bool";
    case "int":
      return t.w === 64 ? "vec2<u32>" : t.s ? "i32" : "u32";
    case "f32":
      return "f32";
    case "ptr":
      return "u32";
    case "arr":
      return `array<${ty_wgsl(t.of)}, ${t.n}>`;
    case "rec":
      return t.union ? "u32" : t.name;
    default:
      return die("a value of type void");
  }
}

// Fold
// ====

// C's integer constant expressions, folded exactly: a value wrapped to its
// type, or null. && and || fold on their first operand alone when it
// decides them, as `0 && x` does.
function fold(u: Unit, n: N): bigint | null {
  const t = n.type ? ty_of(u, n.type) : VOID;
  const xs = n.inner ?? [];
  if (t.k !== "int" && t.k !== "bool") {
    return null;
  }
  switch (n.kind) {
    case "IntegerLiteral":
      return wrap(t, BigInt(n.value));
    case "CXXBoolLiteralExpr":
      return n.value ? 1n : 0n;
    case "ConstantExpr":
      return n.value !== undefined ? wrap(t, BigInt(n.value)) : fold(u, xs[0]);
    case "ParenExpr":
      return fold(u, xs[0]);
    case "ImplicitCastExpr":
    case "CStyleCastExpr": {
      const v = ["IntegralCast", "NoOp", "IntegralToBoolean"]
        .includes(n.castKind) ? fold(u, xs[0]) : null;
      return v === null ? null : wrap(t, v);
    }
    case "UnaryOperator": {
      const v = ["-", "~", "!", "+"].includes(n.opcode) ? fold(u, xs[0])
        : null;
      return v === null ? null : n.opcode === "-" ? wrap(t, -v)
        : n.opcode === "~" ? wrap(t, ~v) : n.opcode === "!" ? (v === 0n
        ? 1n : 0n) : v;
    }
    case "ConditionalOperator": {
      const c = fold(u, xs[0]);
      return c === null ? null : fold(u, xs[c !== 0n ? 1 : 2]);
    }
    case "BinaryOperator": {
      const a = fold(u, xs[0]);
      const op = n.opcode;
      if (op === "&&" || op === "||") {
        const b = a === null || (a !== 0n) === (op === "||") ? null
          : fold(u, xs[1]);
        return a !== null && (a !== 0n) === (op === "||") ? (a !== 0n
          ? 1n : 0n) : b === null ? null : b !== 0n ? 1n : 0n;
      }
      const b = fold(u, xs[1]);
      return a === null || b === null ? null : fold_bin(t, op, a, b);
    }
  }
  return null;
}

function fold_bin(t: Ty, op: string, a: bigint, b: bigint): bigint | null {
  const w = t.k === "int" ? t.w : 1;
  switch (op) {
    case "+": return wrap(t, a + b);
    case "-": return wrap(t, a - b);
    case "*": return wrap(t, a * b);
    case "/": return b === 0n ? null : wrap(t, a / b);
    case "%": return b === 0n ? null : wrap(t, a % b);
    case "&": return wrap(t, a & b);
    case "|": return wrap(t, a | b);
    case "^": return wrap(t, a ^ b);
    case "<<": return b < 0n || b >= BigInt(w) ? null : wrap(t, a << b);
    case ">>": return b < 0n || b >= BigInt(w) ? null : wrap(t, a >> b);
    case "==": return a === b ? 1n : 0n;
    case "!=": return a !== b ? 1n : 0n;
    case "<": return a < b ? 1n : 0n;
    case "<=": return a <= b ? 1n : 0n;
    case ">": return a > b ? 1n : 0n;
    case ">=": return a >= b ? 1n : 0n;
  }
  return null;
}

function wrap(t: Ty, v: bigint): bigint {
  if (t.k !== "int") {
    return v !== 0n ? 1n : 0n;
  }
  const x = BigInt.asUintN(t.w, v);
  return t.s ? BigInt.asIntN(t.w, x) : x;
}

// A literal of an int or bool type.
function lit(t: Ty, v: bigint): Val {
  if (t.k === "bool") {
    return { s: v !== 0n ? "true" : "false", t };
  }
  if (t.k !== "int") {
    return die("a literal of a non-integer type");
  }
  if (t.w === 64) {
    const x = BigInt.asUintN(64, v);
    return { s: `vec2<u32>(${x & 0xFFFFFFFFn}u, ${x >> 32n}u)`, t };
  }
  return { s: !t.s ? `${BigInt.asUintN(32, v)}u` : v === -(2n ** 31n)
    ? "i32(-2147483647 - 1)" : v < 0n ? `(${v}i)` : `${v}i`, t };
}

// Fn
// ==

function fn_new(u: Unit, name: string, ret: Ty): Fn {
  return { u, out: [], ind: "  ", locals: new Map(), names: new Map(),
    tmp: 0, ret, name, alias: new Map() };
}

function emit(f: Fn, line: string): void {
  f.out.push(f.ind + line);
}

function nest(f: Fn, open: string, go: () => void, close = "}"): void {
  emit(f, open);
  const ind = f.ind;
  f.ind += "  ";
  go();
  f.ind = ind;
  emit(f, close);
}

function if_else(f: Fn, c: string, yes: () => void, no: () => void): void {
  nest(f, `if (${c}) {`, yes, "} else {");
  const ind = f.ind;
  f.ind += "  ";
  no();
  f.ind = ind;
  emit(f, "}");
}

// A C name without its leading underscores (WGSL reserves __), one kept
// before a digit (the C emitter's _11_0), counted by what it becomes.
function name_new(f: Fn, base: string): string {
  const stem = base.replace(/^_+/, "").replace(/^(?=\d)/, "_");
  const k = (f.names.get(stem) ?? 0) + 1;
  f.names.set(stem, k);
  return stem + "_" + k;
}

// A value held in a `let`, so what runs after cannot change it.
function hold(f: Fn, v: Val): Val {
  const simple = (s: string): boolean => /^[\w.]*$/.test(s) && !/^[a-z]/i
    .test(s.replace(/^\d+[ui]?$/, "")) || /^vec2<u32>\(\d+u, \d+u\)$/.test(s);
  if (v.p !== undefined) {
    const p = v.p;
    const keep = (s: string): string => simple(s) ? s : hold(f,
      { s, t: U32 }).s;
    return p.pk === "heap" ? { ...v, p: { pk: "heap", ix: keep(p.ix) } }
      : p.pk === "arr" ? { ...v, p: { ...p, off: keep(p.off) } } : v;
  }
  if (simple(v.s)) {
    return v;
  }
  const n = "t" + f.tmp++;
  emit(f, `let ${n} = ${v.s};`);
  return { ...v, s: n };
}

// Val
// ===

// Its value in its type's representation: a bool-held int as 0 or 1.
function val(v: Val): string {
  if (!v.b) {
    return v.s;
  }
  const t = v.t;
  return t.k === "bool" ? v.s : t.k === "int" && t.w === 64
    ? `select(vec2<u32>(0u), vec2<u32>(1u, 0u), ${v.s})`
    : `select(0${int_sfx(t)}, 1${int_sfx(t)}, ${v.s})`;
}

function int_sfx(t: Ty): string {
  return t.k === "int" && t.s ? "i" : "u";
}

// Its truth, as C's scalar test reads it.
function cond(v: Val): string {
  const t = v.t;
  if (v.b || t.k === "bool") {
    return v.s;
  }
  if (v.p !== undefined) {
    return v.p.pk === "heap" ? `(${v.p.ix} != 0u)` : "true";
  }
  return t.k === "f32" ? `(${v.s} != 0.0)` : t.k === "int" && t.w === 64
    ? `any(${v.s} != vec2<u32>(0u))` : `(${v.s} != 0${int_sfx(t)})`;
}

// A literal's value, when the text is one (a folded operand).
function lit_val(v: Val): bigint | null {
  const m = /^\(?(-?\d+)[ui]\)?$/.exec(v.s)
    ?? /^vec2<u32>\((\d+)u, (\d+)u\)$/.exec(v.s);
  return v.b || v.p !== undefined || m === null ? null : m[2] === undefined
    ? BigInt(m[1]) : BigInt(m[1]) + (BigInt(m[2]) << 32n);
}

// An arithmetic value converted to the type t as C converts it.
function conv(v: Val, t: Ty): Val {
  const from = v.t;
  const k = lit_val(v);
  if (k !== null && (t.k === "int" || t.k === "bool") && from.k === "int") {
    return lit(t, wrap(t, from.w === 64 && from.s ? BigInt.asIntN(64, k)
      : k));
  }
  if (t.k === "bool") {
    return { s: cond(v), t };
  }
  if (t.k === "f32") {
    if (from.k === "f32") {
      return { s: v.s, t };
    }
    if (v.b || from.k === "bool") {
      return { s: `select(0.0, 1.0, ${cond(v)})`, t };
    }
    return from.k === "int" && from.w < 64 ? { s: `f32(${v.s})`, t }
      : die("an f32 from a 64-bit integer");
  }
  if (t.k !== "int") {
    return die("a conversion to a non-arithmetic type");
  }
  if (v.b || from.k === "bool") {
    return { s: val({ s: cond(v), t, b: true }), t };
  }
  if (from.k === "f32") {
    return t.w === 64 ? die("a 64-bit integer from an f32")
      : conv({ s: `${t.s ? "i32" : "u32"}(${v.s})`, t: t.s ? I32 : U32 }, t);
  }
  if (from.k !== "int") {
    return die("a conversion from a non-arithmetic type");
  }
  if (t.w === 64) {
    return { s: from.w === 64 ? v.s : from.s ? `i2l(${v.s})`
      : `vec2<u32>(${v.s}, 0u)`, t };
  }
  const s = from.w === 64 ? `${v.s}.x` : v.s;
  const signed = from.w < 64 && from.s;
  const bits = signed ? `bitcast<u32>(${s})` : s;
  if (t.w < 32) {
    const k = 32 - t.w;
    return { s: t.s ? `((bitcast<i32>(${bits}) << ${k}u) >> ${k}u)`
      : `(${bits} & ${2 ** t.w - 1}u)`, t };
  }
  return { s: t.s === signed ? s : t.s ? `bitcast<i32>(${s})` : bits, t };
}

// An int's low 32 bits as a u32: an index, an offset, a shift count.
function low(v: Val): string {
  return conv(v, U32).s;
}

// An f32 by its bits; WGSL refuses a constant infinity or NaN, so those
// read NZ, a zero no constant can see through.
function f32_lit(x: number): string {
  const d = new DataView(new ArrayBuffer(4));
  d.setFloat32(0, x);
  return `bitcast<f32>(0x${d.getUint32(0).toString(16)}u${Number.isFinite(x)
    ? "" : " | NZ"})`;
}

// Index arithmetic, folded when both sides are literals.
function ix_add(a: string, b: string): string {
  const x = /^(\d+)u$/.exec(a);
  const y = /^(\d+)u$/.exec(b);
  return x && y ? `${(Number(x[1]) + Number(y[1])) >>> 0}u` : y?.[1] === "0"
    ? a : x?.[1] === "0" ? b : `(${a} + ${b})`;
}

function ix_mul(a: string, k: number): string {
  const x = /^(\d+)u$/.exec(a);
  return x ? `${Math.imul(Number(x[1]), k) >>> 0}u` : k === 1 ? a
    : `(${a} * ${k}u)`;
}

// Place
// =====

function place(f: Fn, n: N): Place {
  const t = ty_of(f.u, n.type);
  const xs = n.inner ?? [];
  switch (n.kind) {
    case "ParenExpr":
      return place(f, xs[0]);
    case "DeclRefExpr": {
      const d = n.referencedDecl;
      const l = f.locals.get(d.id);
      if (l !== undefined) {
        return { lk: "ref", r: l.name, t };
      }
      if (f.u.globs.has(d.name)) {
        return { lk: "tab", at: `${tab_of(f.u, d.name)}u`, t };
      }
      return die("a reference it cannot see: " + d.name);
    }
    case "ArraySubscriptExpr": {
      const b = ex(f, xs[0]);
      const i = ex(f, xs[1]);
      return b.p !== undefined ? elem(b.p, i, t)
        : die("a subscript whose base is not a pointer");
    }
    case "UnaryOperator": {
      const p = n.opcode === "*" ? ex(f, xs[0]).p : undefined;
      return p !== undefined ? elem(p, lit(U32, 0n), t)
        : die("a place of a unary " + n.opcode);
    }
    case "MemberExpr": {
      const base = n.isArrow ? null : place(f, xs[0]);
      const rt = n.isArrow ? ptr_to(ex(f, xs[0])) : base!.t;
      const fld = rt.k === "rec" ? rt.fs.find((x) => x.name === n.name)
        : undefined;
      if (fld === undefined || rt.k !== "rec") {
        return die("a member it cannot find: " + n.name);
      }
      if (n.isArrow) {
        const p = ex(f, xs[0]).p!;
        return p.pk === "heap" ? { lk: "heap", ix: ix_add(p.ix,
          `${fld.off / 4}u`), t } : die("a member through a local pointer");
      }
      const b = base!;
      return rt.union ? (b.lk === "ref" ? { lk: "bits", r: b.r, t }
        : die("a union outside a local"))
        : b.lk === "ref" ? { lk: "ref", r: `${b.r}.f_${n.name}`, t }
        : b.lk === "heap" ? { lk: "heap", ix: ix_add(b.ix, `${fld.off / 4}u`),
          t } : die("a member of a table");
    }
  }
  return die("a place of " + n.kind);
}

function ptr_to(v: Val): Ty {
  return v.t.k === "ptr" ? v.t.to : die("a -> on a non-pointer");
}

// The element i past where p points.
function elem(p: Ptr, i: Val, t: Ty): Place {
  const k = low(i);
  switch (p.pk) {
    case "heap":
      return { lk: "heap", ix: ix_add(p.ix, ix_mul(k, ty_words(t))), t };
    case "ref":
      return k === "0u" ? { lk: "ref", r: p.r, t }
        : die("an offset from a pointer to a scalar");
    case "arr":
      return { lk: "ref", r: `${p.r}[${ix_add(p.off, k)}]`, t };
    case "tab":
      return { lk: "tab", at: ix_add(`${p.at}u`, ix_mul(k, t.k === "int"
        && t.w === 64 ? 2 : 1)), t };
  }
}

function addr(pl: Place): Ptr {
  switch (pl.lk) {
    case "ref":
      return { pk: "ref", r: pl.r };
    case "heap":
      return { pk: "heap", ix: pl.ix };
    case "tab":
      return { pk: "tab", at: Number(/^(\d+)u$/.exec(pl.at)?.[1] ?? die(
        "a table address at a moving index")) };
    default:
      return die("the address of a union member");
  }
}

function load(f: Fn, pl: Place): Val {
  const t = pl.t;
  if (t.k === "ptr") {
    return pl.lk === "ref" ? { s: "", t, p: f.alias.get(pl.r)
      ?? { pk: "heap", ix: pl.r } } : die("a pointer read from memory");
  }
  switch (pl.lk) {
    case "ref":
      return { s: pl.r, t };
    case "bits":
      return { s: t.k === "f32" ? `bitcast<f32>(${pl.r})` : t.k === "int"
        && t.s ? `bitcast<i32>(${pl.r})` : pl.r, t };
    case "tab":
      return { s: t.k === "int" && t.w === 64
        ? `vec2<u32>(TAB[${pl.at}], TAB[${ix_add(pl.at, "1u")}])`
        : t.k === "f32" ? `bitcast<f32>(TAB[${pl.at}])` : t.k === "int"
        && t.s ? `bitcast<i32>(TAB[${pl.at}])` : `TAB[${pl.at}]`, t };
    case "heap":
      return t.k === "int" && t.w === 64 ? { s: `ld64(${pl.ix})`, t }
        : t.k === "int" && t.w === 32 ? { s: t.s
          ? `bitcast<i32>(atomicLoad(&M[${pl.ix}]))`
          : `atomicLoad(&M[${pl.ix}])`, t }
        : t.k === "f32" ? { s: `bitcast<f32>(atomicLoad(&M[${pl.ix}]))`, t }
        : die("a corpus load of a " + t.k);
  }
}

function store(f: Fn, pl: Place, v: Val): void {
  const t = pl.t;
  if (t.k === "ptr") {
    if (pl.lk !== "ref" || f.alias.has(pl.r) || v.p?.pk !== "heap") {
      die("a pointer stored somewhere other than a corpus index variable");
    }
    emit(f, `${pl.r} = ${v.p.ix};`);
    return;
  }
  const x = val(conv_as(v, t));
  switch (pl.lk) {
    case "ref":
      emit(f, `${pl.r} = ${x};`);
      return;
    case "bits":
      emit(f, `${pl.r} = ${t.k === "f32" || t.k === "int" && t.s
        ? `bitcast<u32>(${x})` : x};`);
      return;
    case "heap":
      emit(f, t.k === "int" && t.w === 64 ? `st64(${pl.ix}, ${x});`
        : t.k === "int" && t.w === 32 || t.k === "f32"
        ? `st32(${pl.ix}, ${t.k === "int" && !t.s ? x
          : `bitcast<u32>(${x})`});` : die("a corpus store of a " + t.k));
      return;
    default:
      die("a store into a table");
  }
}

// A value made the type t: arithmetic converted, a record or a pointer as
// it is.
function conv_as(v: Val, t: Ty): Val {
  return (t.k === "int" || t.k === "bool" || t.k === "f32")
    && (v.t.k === "int" || v.t.k === "bool" || v.t.k === "f32") ? conv(v, t)
    : v;
}

// Tab
// ===

// A global table's first word in TAB, its initializer folded: a u64 row
// is two words, an f32 its bits.
function tab_of(u: Unit, name: string): number {
  const hit = u.tabs.get(name);
  if (hit !== undefined) {
    return hit;
  }
  const d = u.globs.get(name)!;
  const t = ty_of(u, d.type);
  const el = t.k === "arr" ? t.of : die("a global that is not a table: "
    + name);
  const at = u.tab.length;
  const init = d.inner?.[0]?.kind === "InitListExpr" ? d.inner[0].inner ?? []
    : die("a table without an initializer: " + name);
  for (const x of init) {
    const v = el.k === "f32" ? null : fold(u, x);
    if (el.k === "f32") {
      const d = new DataView(new ArrayBuffer(4));
      d.setFloat32(0, f32_fold(x));
      u.tab.push(d.getUint32(0));
    } else if (v === null) {
      die("a table row that does not fold: " + name);
    } else if (el.k === "int" && el.w === 64) {
      const w = BigInt.asUintN(64, v);
      u.tab.push(Number(w & 0xFFFFFFFFn), Number(w >> 32n));
    } else {
      u.tab.push(Number(BigInt.asUintN(32, v)));
    }
  }
  u.tabs.set(name, at);
  return at;
}

function f32_fold(n: N): number {
  return n.kind === "FloatingLiteral" ? Number(n.value) : n.inner?.length
    ? f32_fold(n.inner[0]) : die("an f32 table row that does not fold");
}

// Ex
// ==

function ex(f: Fn, n: N): Val {
  const t = n.type ? ty_of(f.u, n.type) : VOID;
  const xs = n.inner ?? [];
  if (t.k === "int" || t.k === "bool") {
    const c = fold(f.u, n);
    if (c !== null) {
      return lit(t, c);
    }
  }
  switch (n.kind) {
    case "ParenExpr":
    case "ConstantExpr":
      return ex(f, xs[0]);
    case "FloatingLiteral":
      return { s: f32_lit(Number(n.value)), t: F32 };
    case "ImplicitCastExpr":
    case "CStyleCastExpr":
      return ex_cast(f, n, t);
    case "BinaryOperator":
      return ex_bin(f, n, t);
    case "CompoundAssignOperator":
      return ex_set(f, n, true);
    case "UnaryOperator":
      return ex_un(f, n, t);
    case "ConditionalOperator":
      return ex_sel(f, n, t);
    case "CallExpr":
      return ex_call(f, n, t);
  }
  return die("an expression it does not know: " + n.kind + " in " + f.name);
}

function ex_cast(f: Fn, n: N, t: Ty): Val {
  const x = n.inner![0];
  switch (n.castKind) {
    case "LValueToRValue":
      return load(f, place(f, x));
    case "NoOp":
    case "FloatingCast":
      return { ...ex(f, x), t };
    case "IntegralCast":
    case "IntegralToBoolean":
    case "FloatingToBoolean":
    case "IntegralToFloating":
    case "FloatingToIntegral":
    case "BooleanToSignedIntegral":
      return conv(ex(f, x), t);
    case "PointerToBoolean":
      return { s: cond(ex(f, x)), t: BOOL };
    case "BitCast": {
      const v = ex(f, x);
      return v.p !== undefined ? { s: "", t, p: v.p }
        : die("a bit cast of a non-pointer");
    }
    case "ArrayToPointerDecay": {
      const pl = place(f, x);
      const p: Ptr = pl.lk === "tab" ? addr(pl) : pl.lk === "ref"
        && pl.t.k === "arr" ? { pk: "arr", r: pl.r, off: "0u", n: pl.t.n }
        : die("an array that is neither local nor a table");
      return { s: "", t, p };
    }
    case "NullToPointer":
      return { s: "", t, p: { pk: "heap", ix: "0u" } };
    case "ToVoid":
      return { ...ex(f, x), t: VOID };
  }
  return die("a cast it does not know: " + n.castKind);
}

const CMPS = ["==", "!=", "<", "<=", ">", ">="];

function ex_bin(f: Fn, n: N, t: Ty): Val {
  const op = n.opcode;
  const [l, r] = n.inner!;
  if (op === "=") {
    return ex_set(f, n, false);
  }
  if (op === ",") {
    stmt(f, l);
    return ex(f, r);
  }
  if (op === "&&" || op === "||") {
    const k = fold(f.u, l);
    if (k !== null) {
      return { s: cond(ex(f, r)), t, b: true };
    }
    const a = cond(ex(f, l));
    if (!needs(f.u, r)) {
      return { s: `(${a} ${op} ${cond(ex(f, r))})`, t, b: true };
    }
    const v = "t" + f.tmp++;
    emit(f, `var ${v} = ${a};`);
    nest(f, `if (${op === "&&" ? v : "!" + v}) {`, () => emit(f,
      `${v} = ${cond(ex(f, r))};`));
    return { s: v, t, b: true };
  }
  let a = ex(f, l);
  if (needs(f.u, r)) {
    a = hold(f, a);
  }
  const b = ex(f, r);
  if (a.p !== undefined || b.p !== undefined) {
    return ex_ptr(f, op, a, b, t);
  }
  return CMPS.includes(op) ? { s: cmp(op, a, b), t, b: true }
    : arith(op, a, b, t);
}

function cmp(op: string, a: Val, b: Val): string {
  const t = a.b ? b.t : a.t;
  const A = val(a);
  const B = val(b);
  if (t.k !== "int" || t.w !== 64) {
    return `(${A} ${op} ${B})`;
  }
  const lt = t.s ? "i64_lt" : "u64_lt";
  switch (op) {
    case "==": return `all(${A} == ${B})`;
    case "!=": return `any(${A} != ${B})`;
    case "<": return `${lt}(${A}, ${B})`;
    case ">": return `${lt}(${B}, ${A})`;
    case "<=": return `!${lt}(${B}, ${A})`;
    default: return `!${lt}(${A}, ${B})`;
  }
}

function arith(op: string, a: Val, b: Val, t: Ty): Val {
  const A = val(conv_as(a, t));
  if (op === "<<" || op === ">>") {
    const k = low(b);
    if (t.k === "int" && t.w === 64) {
      return { s: `${op === "<<" ? "u64_shl" : t.s ? "i64_shr" : "u64_shr"}`
        + `(${A}, ${k})`, t };
    }
    return { s: `(${A} ${op} ${k})`, t };
  }
  const B = val(conv_as(b, t));
  if (t.k === "int" && t.w < 64 && !t.s && (op === "/" || op === "%")) {
    const k = lit_val(conv_as(b, t));
    return k !== null && k > 0n && (k & (k - 1n)) === 0n ? { s: op === "/"
      ? `(${A} >> ${k.toString(2).length - 1}u)` : `(${A} & ${k - 1n}u)`, t }
      : { s: `${op === "/" ? "u32_div" : "u32_rem"}(${A}, ${B})`, t };
  }
  if (t.k === "int" && t.w === 64) {
    const fn: Record<string, string> = { "+": "u64_add", "-": "u64_sub",
      "*": "u64_mul", "/": t.s ? "i64_div" : "u64_div", "%": t.s ? "i64_rem"
      : "u64_rem" };
    return fn[op] !== undefined ? { s: `${fn[op]}(${A}, ${B})`, t }
      : { s: `(${A} ${op} ${B})`, t };
  }
  return { s: `(${A} ${op} ${B})`, t };
}

// Pointer arithmetic and comparison: an offset scales by the element's
// words; two pointers compare as corpus indices.
function ex_ptr(f: Fn, op: string, a: Val, b: Val, t: Ty): Val {
  if (a.p !== undefined && b.p !== undefined) {
    return CMPS.includes(op) && a.p.pk === "heap" && b.p.pk === "heap"
      ? { s: `(${a.p.ix} ${op} ${b.p.ix})`, t, b: true }
      : die("a pointer difference, or a comparison off the corpus");
  }
  const [p, i] = a.p !== undefined ? [a, b] : [b, a];
  if (op !== "+" && !(op === "-" && a.p !== undefined)) {
    return die("a pointer operator " + op);
  }
  return { s: "", t, p: ptr_move(ptr_to(p), p.p!, i, op === "-") };
}

function ptr_move(el: Ty, p: Ptr, i: Val, back: boolean): Ptr {
  const k = low(i);
  const d = back ? (/^\d+u$/.test(k) ? `${(-Number(k.slice(0, -1))) >>> 0}u`
    : `(0u - ${k})`) : k;
  switch (p.pk) {
    case "heap":
      return { pk: "heap", ix: ix_add(p.ix, ix_mul(d, ty_words(el))) };
    case "arr":
      return { ...p, off: ix_add(p.off, d) };
    case "tab":
      return /^\d+u$/.test(d) ? { pk: "tab", at: p.at + Number(d.slice(0, -1))
        * (el.k === "int" && el.w === 64 ? 2 : 1) }
        : die("a table pointer moved by a variable");
    default:
      return k === "0u" ? p : die("a pointer to a scalar moved");
  }
}

// An assignment, plain or compound; its value is what was stored.
function ex_set(f: Fn, n: N, compound: boolean): Val {
  const [l, r] = n.inner!;
  let pl = place(f, l);
  if (needs(f.u, r) && pl.lk === "heap") {
    pl = { ...pl, ix: hold(f, { s: pl.ix, t: U32 }).s };
  }
  const v = hold(f, compound ? set_op(f, n, pl, r) : conv_as(ex(f, r), pl.t));
  store(f, pl, v);
  return v.p !== undefined ? v : { ...v, s: val(v), b: false, t: pl.t };
}

function set_op(f: Fn, n: N, pl: Place, r: N): Val {
  const op = n.opcode.slice(0, -1);
  const old = load(f, pl);
  if (old.p !== undefined) {
    return { s: "", t: pl.t, p: ptr_move(ptr_to(old), old.p, ex(f, r),
      op === "-") };
  }
  const lt = ty_of(f.u, n.computeLHSType);
  const rt = ty_of(f.u, n.computeResultType);
  return conv(arith(op, conv(old, lt), ex(f, r), rt), pl.t);
}

function ex_un(f: Fn, n: N, t: Ty): Val {
  const x = n.inner![0];
  switch (n.opcode) {
    case "&":
      return { s: "", t, p: addr(place(f, x)) };
    case "++":
    case "--":
      return ex_step(f, n);
    case "!":
      return { s: `!${cond(ex(f, x))}`, t, b: true };
    case "+":
      return ex(f, x);
    case "~": {
      const v = val(ex(f, x));
      return { s: `(~${v})`, t };
    }
    case "-": {
      const v = val(ex(f, x));
      return { s: t.k === "int" && t.w === 64 ? `u64_sub(vec2<u32>(0u), ${v})`
        : t.k === "int" && !t.s ? `(0u - ${v})` : `(-${v})`, t };
    }
  }
  return die("a unary operator " + n.opcode);
}

// ++ and --: the new value stored; the expression is the new or the old.
function ex_step(f: Fn, n: N): Val {
  const pl0 = place(f, n.inner![0]);
  const pl = pl0.lk === "heap" ? { ...pl0, ix: hold(f, { s: pl0.ix, t: U32 })
    .s } : pl0;
  const old = hold(f, load(f, pl));
  const one = lit(pl.t.k === "int" ? pl.t : I32, 1n);
  const op = n.opcode === "++" ? "+" : "-";
  const nw = old.p !== undefined ? { s: "", t: pl.t, p: ptr_move(ptr_to(old),
    old.p, one, op === "-") } : hold(f, pl.t.k === "f32" ? { s: `(${old.s} `
    + `${op} 1.0)`, t: F32 } : arith(op, old, one, pl.t));
  store(f, pl, nw);
  return n.isPostfix ? old : nw;
}

// c ? a : b: a select when both sides are light and pure, else an if.
function ex_sel(f: Fn, n: N, t: Ty): Val {
  const [c, a, b] = n.inner!;
  const k = fold(f.u, c);
  if (k !== null) {
    return ex(f, k !== 0n ? a : b);
  }
  if (!needs(f.u, n)) {
    const C = cond(ex(f, c));
    const A = conv_as(ex(f, a), t);
    const B = conv_as(ex(f, b), t);
    if (A.p !== undefined || B.p !== undefined) {
      return A.p?.pk === "heap" && B.p?.pk === "heap" ? { s: "", t, p: {
        pk: "heap", ix: `select(${B.p.ix}, ${A.p.ix}, ${C})` } }
        : die("a choice between pointers off the corpus");
    }
    return { s: `select(${val(B)}, ${val(A)}, ${C})`, t };
  }
  const C = cond(ex(f, c));
  const v = "t" + f.tmp++;
  emit(f, `var ${v}: ${ty_wgsl(t)};`);
  const arm = (x: N): void => {
    const y = conv_as(ex(f, x), t);
    emit(f, `${v} = ${y.p !== undefined ? (y.p.pk === "heap" ? y.p.ix
      : die("a chosen pointer off the corpus")) : val(y)};`);
  };
  if_else(f, C, () => arm(a), () => arm(b));
  return t.k === "ptr" ? { s: "", t, p: { pk: "heap", ix: v } } : { s: v, t };
}

// Needs
// =====

// Does an expression need statements before it: an assignment, a step, a
// comma, a short-circuit whose second side does, or a choice that cannot
// be a select (a side neither light nor pure)?
function needs(u: Unit, n: N): boolean {
  switch (n.kind) {
    case "CompoundAssignOperator":
      return true;
    case "UnaryOperator":
      return n.opcode === "++" || n.opcode === "--" || needs(u, n.inner![0]);
    case "BinaryOperator":
      if (n.opcode === "=" || n.opcode === ",") {
        return true;
      }
      break;
    case "ConditionalOperator": {
      const [c, a, b] = n.inner!;
      const k = fold(u, c);
      return k !== null ? needs(u, k !== 0n ? a : b) : needs(u, c)
        || !light(u, a) || !light(u, b);
    }
  }
  return (n.inner ?? []).some((x) => needs(u, x));
}

// A light expression calls only light functions (pure, loop-free) and
// writes nothing: a select may run it for nothing.
function light(u: Unit, n: N): boolean {
  if (n.kind === "CallExpr") {
    const name = callee(n);
    if (!(MATH1.includes(name) || MATH2.includes(name) || name === "__clz"
      || fn_light(u, name))) {
      return false;
    }
  }
  return !needs(u, n) && (n.inner ?? []).every((x) => light(u, x));
}

function fn_light(u: Unit, name: string): boolean {
  const hit = u.pure.get(name);
  if (hit !== undefined) {
    return hit;
  }
  u.pure.set(name, false);
  const d = u.fns.get(name);
  const ok = (n: N): boolean => !["ForStmt", "WhileStmt", "DoStmt",
    "SwitchStmt"].includes(n.kind) && !(n.kind === "CallExpr"
    && !light(u, n)) && !((n.kind === "BinaryOperator" && n.opcode === "=")
    || n.kind === "CompoundAssignOperator" || (n.kind === "UnaryOperator"
    && (n.opcode === "++" || n.opcode === "--"))) && (n.inner ?? []).every(ok);
  const res = d !== undefined && (d.inner ?? []).every((x) => x.kind
    === "ParmVarDecl" || ok(x));
  u.pure.set(name, res);
  return res;
}

function callee(n: N): string {
  let c = n.inner![0];
  while (c.kind !== "DeclRefExpr") {
    c = c.inner?.[0] ?? die("a call through a pointer");
  }
  return c.referencedDecl.name;
}

// Call
// ====

// Arguments in order; one that needs statements first holds those before.
function ex_list(f: Fn, xs: N[]): Val[] {
  const vs: Val[] = [];
  for (const x of xs) {
    if (needs(f.u, x)) {
      vs.forEach((v, j) => vs[j] = hold(f, v));
    }
    vs.push(ex(f, x));
  }
  return vs;
}

function ex_call(f: Fn, n: N, t: Ty): Val {
  const name = callee(n);
  const vs = ex_list(f, n.inner!.slice(1));
  const heap = (v: Val): string => v.p?.pk === "heap" ? v.p.ix
    : die("an atomic off the corpus");
  if (name === "atomicCAS") {
    return { s: `c_cas(${heap(vs[0])}, ${val(vs[1])}, ${val(vs[2])})`, t };
  }
  if (name.startsWith("atomic")) {
    return { s: `${name}(&M[${heap(vs[0])}], ${val(vs[1])})`, t };
  }
  if (name === "__threadfence") {
    return { s: "", t };
  }
  if (name === "__clz") {
    return { s: `countLeadingZeros(${val(vs[0])})`, t };
  }
  if (MATH1.includes(name) || MATH2.includes(name)) {
    const w = { fabs: "abs", log10: "c_log10", pow: "c_pow", atan2: "c_atan2",
      fmod: "c_fmod" }[name] ?? name;
    return { s: `${w}(${vs.map((v) => val(conv(v, F32))).join(", ")})`, t };
  }
  if (!f.u.fns.has(name)) {
    return die("a call to " + name + ", which the device does not hold");
  }
  const inst = inst_of(f.u, name, vs.map((v) => v.p));
  const s = `${inst}(${vs.flatMap((v) => v.p === undefined ? [val(v)]
    : v.p.pk === "heap" ? [v.p.ix] : v.p.pk === "ref" ? [`&(${v.p.r})`]
    : v.p.pk === "arr" ? [`&(${v.p.r})`, v.p.off]
    : die("a table passed to " + name)).join(", ")})`;
  return t.k === "ptr" ? { s: "", t, p: { pk: "heap", ix: s } } : { s, t };
}

// A function's instance for where its pointer arguments point: a pointer
// into the corpus is an index, one at a local a WGSL pointer.
function inst_of(u: Unit, name: string, ps: (Ptr | undefined)[]): string {
  const shape = ps.map((p) => p === undefined || p.pk === "heap" ? "h"
    : p.pk === "ref" ? "r" : p.pk === "arr" ? "a" + p.n
    : die("a table passed to " + name));
  const key = name + "|" + shape.join("");
  const hit = u.done.get(key);
  if (hit !== undefined) {
    return hit;
  }
  const inst = "c_" + name + (shape.some((s) => s !== "h") ? "_"
    + shape.join("") : "");
  u.done.set(key, inst);
  u.todo.push([inst, u.fns.get(name)!, ps as Ptr[]]);
  return inst;
}

// Stmt
// ====

function stmt(f: Fn, n: N): void {
  const xs = n.inner ?? [];
  switch (n.kind) {
    case "CompoundStmt":
      nest(f, "{", () => xs.forEach((x) => stmt(f, x)));
      return;
    case "DeclStmt":
      xs.forEach((x) => decl(f, x));
      return;
    case "NullStmt":
      return;
    case "BreakStmt":
      emit(f, "break;");
      return;
    case "ContinueStmt":
      emit(f, "continue;");
      return;
    case "ReturnStmt":
      return st_ret(f, xs[0]);
    case "IfStmt":
      return st_if(f, n);
    case "ForStmt":
      return st_loop(f, xs[0], xs[2], xs[3], xs[4]);
    case "WhileStmt":
      return st_loop(f, undefined, xs[xs.length - 2], undefined,
        xs[xs.length - 1]);
    case "DoStmt":
      return st_do(f, xs[0], xs[1]);
    case "SwitchStmt":
      return st_switch(f, n);
  }
  st_expr(f, n);
}

function body(f: Fn, n: N): void {
  (n.kind === "CompoundStmt" ? n.inner ?? [] : [n]).forEach((x) => stmt(f,
    x));
}

// An expression for its effects: no value kept.
function st_expr(f: Fn, n: N): void {
  const xs = n.inner ?? [];
  switch (n.kind) {
    case "ParenExpr":
      return st_expr(f, xs[0]);
    case "ImplicitCastExpr":
    case "CStyleCastExpr":
      if (n.castKind === "ToVoid") {
        return st_expr(f, xs[0]);
      }
      break;
    case "BinaryOperator":
      if (n.opcode === ",") {
        st_expr(f, xs[0]);
        return st_expr(f, xs[1]);
      }
      if (n.opcode === "=") {
        const pl0 = place(f, xs[0]);
        const pl = needs(f.u, xs[1]) && pl0.lk === "heap" ? { ...pl0,
          ix: hold(f, { s: pl0.ix, t: U32 }).s } : pl0;
        return store(f, pl, conv_as(ex(f, xs[1]), pl.t));
      }
      break;
    case "CompoundAssignOperator": {
      const pl = place(f, xs[0]);
      return store(f, pl, set_op(f, n, pl, xs[1]));
    }
    case "UnaryOperator":
      if (n.opcode === "++" || n.opcode === "--") {
        ex_step(f, n);
        return;
      }
      break;
    case "ConditionalOperator": {
      const k = fold(f.u, xs[0]);
      if (k !== null) {
        return st_expr(f, xs[k !== 0n ? 1 : 2]);
      }
      return if_else(f, cond(ex(f, xs[0])), () => st_expr(f, xs[1]),
        () => st_expr(f, xs[2]));
    }
    case "CallExpr": {
      const v = ex(f, n);
      const s = v.p?.pk === "heap" ? v.p.ix : v.s;
      if (/^(c_|atomic)\w*\(/.test(s)) {
        emit(f, s + ";");
      }
      return;
    }
  }
  const v = ex(f, n);
  if (/\bc_\w+\(|\batomic\w+\(/.test(v.s)) {
    emit(f, `_ = ${val(v)};`);
  }
}

function st_ret(f: Fn, x?: N): void {
  if (x === undefined) {
    return emit(f, "return;");
  }
  if (f.ret.k === "void") {
    st_expr(f, x);
    return emit(f, "return;");
  }
  const v = conv_as(ex(f, x), f.ret);
  emit(f, `return ${v.p === undefined ? val(v) : v.p.pk === "heap" ? v.p.ix
    : die("a pointer returned off the corpus")};`);
}

function st_if(f: Fn, n: N): void {
  const [c, yes, no] = n.inner!;
  const k = fold(f.u, c);
  if (k !== null) {
    const b = k !== 0n ? yes : n.hasElse ? no : undefined;
    if (b !== undefined) {
      nest(f, "{", () => body(f, b));
    }
    return;
  }
  const C = cond(ex(f, c));
  if (n.hasElse) {
    return if_else(f, C, () => body(f, yes), () => body(f, no));
  }
  nest(f, `if (${C}) {`, () => body(f, yes));
}

// for, while: a loop whose test breaks and whose step is its continuing,
// where a continue lands, as in C.
function st_loop(f: Fn, init: N | undefined, c: N | undefined,
  step: N | undefined, b: N): void {
  const has = (x?: N): x is N => x?.kind !== undefined;
  nest(f, "{", () => {
    if (has(init)) {
      stmt(f, init);
    }
    const k = has(c) ? fold(f.u, c) : 1n;
    if (k === 0n) {
      return;
    }
    nest(f, "loop {", () => {
      if (k === null) {
        emit(f, `if (!(${cond(ex(f, c!))})) { break; }`);
      }
      nest(f, "{", () => body(f, b));
      if (has(step)) {
        nest(f, "continuing {", () => st_expr(f, step));
      }
    });
  });
}

function st_do(f: Fn, b: N, c: N): void {
  const k = fold(f.u, c);
  nest(f, "loop {", () => {
    nest(f, "{", () => body(f, b));
    if (k === null || k === 0n) {
      nest(f, "continuing {", () => emit(f, `break if !(${k === 0n ? "false"
        : cond(ex(f, c))});`));
    }
  });
}

// A switch: labels grouped, and a group that can fall through runs the
// next group's statements after its own, as C falls into them.
function st_switch(f: Fn, n: N): void {
  const [c, b] = n.inner!;
  const sel = ex(f, c);
  const t = sel.t;
  if (t.k !== "int" || t.w === 64) {
    return die("a switch on a " + t.k);
  }
  const groups: { labels: string[]; body: N[] }[] = [];
  let cur: { labels: string[]; body: N[] } | null = null;
  for (let s of b.inner ?? []) {
    while (s.kind === "CaseStmt" || s.kind === "DefaultStmt") {
      if (cur === null || cur.body.length > 0) {
        cur = { labels: [], body: [] };
        groups.push(cur);
      }
      cur.labels.push(s.kind === "DefaultStmt" ? "default" : lit(t,
        fold(f.u, s.inner![0]) ?? die("a case that does not fold")).s);
      s = s.inner![s.inner!.length - 1];
    }
    (cur ?? die("a statement before a switch's first case")).body.push(s);
  }
  nest(f, `switch (${val(sel)}) {`, () => {
    groups.forEach((g, i) => nest(f, `case ${g.labels.join(", ")}: {`, () => {
      for (let j = i; j < groups.length; j += 1) {
        groups[j].body.forEach((x) => stmt(f, x));
        if (!falls(groups[j].body)) {
          break;
        }
      }
    }));
    if (!groups.some((g) => g.labels.includes("default"))) {
      emit(f, "default: {}");
    }
  });
}

// Can control run past the last statement?
function falls(xs: N[]): boolean {
  const ys = xs.filter((x) => x.kind !== "NullStmt");
  const last = ys[ys.length - 1];
  switch (last?.kind) {
    case "BreakStmt":
    case "ContinueStmt":
    case "ReturnStmt":
      return false;
    case "CompoundStmt":
      return falls(last.inner ?? []);
    case "IfStmt":
      return !last.hasElse || falls([last.inner![1]])
        || falls([last.inner![2]]);
  }
  return true;
}

// Decl
// ====

function decl(f: Fn, d: N): void {
  if (d.kind === "RecordDecl" || d.kind === "TypedefDecl") {
    return;
  }
  if (d.kind !== "VarDecl" || d.storageClass === "static") {
    return die("a local declaration it does not know in " + f.name);
  }
  const t = ty_of(f.u, d.type);
  const init = d.init !== undefined ? d.inner![0] : undefined;
  if (t.k === "ptr") {
    const p = init !== undefined ? ex(f, init).p
      ?? die("a pointer from a non-pointer") : { pk: "heap", ix: "0u" } as Ptr;
    const name = name_new(f, d.name);
    f.locals.set(d.id, { name, t });
    if (p.pk === "heap") {
      emit(f, `var ${name}: u32 = ${p.ix};`);
    } else {
      f.alias.set(name, hold(f, { s: "", t, p }).p!);
    }
    return;
  }
  const x = init === undefined ? "" : " = " + (init.kind === "InitListExpr"
    ? init_list(f, t, init) : val(conv_as(ex(f, init), t)));
  const name = name_new(f, d.name);
  f.locals.set(d.id, { name, t });
  emit(f, `var ${name}: ${ty_wgsl(t)}${x};`);
}

function init_list(f: Fn, t: Ty, n: N): string {
  const vs = ex_list(f, n.inner ?? []);
  const as = (v: Val, ft: Ty): string => v.p === undefined
    ? val(conv_as(v, ft)) : v.p.pk === "heap" ? v.p.ix
    : die("a pointer off the corpus in a record");
  if (t.k === "rec" && t.union) {
    const ft = t.fs.find((x) => x.name === n.field?.name)?.t
      ?? die("a union initialized by a member it cannot find");
    const x = as(vs[0], ft);
    return ft.k === "f32" || ft.k === "int" && ft.s ? `bitcast<u32>(${x})` : x;
  }
  if (t.k === "rec") {
    return `${ty_wgsl(t)}(${t.fs.map((fl, i) => i < vs.length ? as(vs[i],
      fl.t) : zero(fl.t)).join(", ")})`;
  }
  if (t.k === "arr") {
    return `${ty_wgsl(t)}(${[...Array(t.n)].map((_, i) => i < vs.length
      ? as(vs[i], t.of) : zero(t.of)).join(", ")})`;
  }
  return die("an initializer list for a " + t.k);
}

function zero(t: Ty): string {
  return t.k === "bool" ? "false" : t.k === "f32" ? "0.0" : t.k === "ptr"
    ? "0u" : t.k === "int" ? (t.w === 64 ? "vec2<u32>(0u)" : t.s ? "0i" : "0u")
    : `${ty_wgsl(t)}()`;
}

// Emit
// ====

function fn_emit(u: Unit, inst: string, d: N, ps: (Ptr | undefined)[])
  : string {
  const ft: string = d.type.qualType;
  const f = fn_new(u, inst, ty_str(u, ft.slice(0, ft.indexOf("(")).trim()));
  const params: string[] = [];
  const copies: string[] = [];
  d.inner!.filter((x) => x.kind === "ParmVarDecl").forEach((p, i) => {
    const t = ty_of(u, p.type);
    const name = name_new(f, p.name ?? "p");
    const sh = ps[i];
    f.locals.set(p.id, { name, t });
    if (t.k === "ptr" && sh !== undefined && sh.pk !== "heap") {
      const el = ty_wgsl(t.to);
      if (sh.pk === "ref") {
        params.push(`${name}: ptr<function, ${el}>`);
        f.alias.set(name, { pk: "ref", r: `(*${name})` });
      } else if (sh.pk === "arr") {
        params.push(`${name}: ptr<function, array<${el}, ${sh.n}>>`,
          `${name}_o: u32`);
        f.alias.set(name, { pk: "arr", r: `(*${name})`, off: `${name}_o`,
          n: sh.n });
      }
      return;
    }
    params.push(`${name}_a: ${ty_wgsl(t)}`);
    copies.push(`  var ${name} = ${name}_a;`);
  });
  const b = d.inner!.find((x) => x.kind === "CompoundStmt")!;
  body(f, b);
  const last = (b.inner ?? []).filter((x) => x.kind !== "NullStmt").pop();
  if (f.ret.k !== "void" && last?.kind !== "ReturnStmt") {
    emit(f, `return ${zero(f.ret)};`);
  }
  return `fn ${inst}(${params.join(", ")})${f.ret.k === "void" ? ""
    : " -> " + ty_wgsl(f.ret)} {\n${[...copies, ...f.out].join("\n")}\n}\n`;
}

// The device program: every function the rounds reach and the records
// they pass by value, and TAB's words.
function device_of(ast: N): { code: string; tab: number[] } {
  const u = unit_new(ast);
  inst_of(u, "wg_plan", []);
  inst_of(u, "wg_packing", []);
  inst_of(u, "wg_run", []);
  inst_of(u, "wg_packs", []);
  inst_of(u, "wg_window", []);
  const fns: string[] = [];
  while (u.todo.length > 0) {
    const [inst, d, ps] = u.todo.pop()!;
    fns.push(fn_emit(u, inst, d, ps));
  }
  const recs = [...u.recs.keys()].map((id) => ty_rec(u, id)).filter((t) =>
    t.k === "rec" && !t.union && t.fs.length > 0).map((t) => t.k === "rec"
    ? `struct ${ty_wgsl(t)} { ${t.fs.map((x) => `f_${x.name}: `
      + ty_wgsl(x.t)).join(", ")} }\n` : "");
  return { code: recs.join("") + fns.join("\n"), tab: u.tab };
}

// Helpers
// =======

// What the translated C calls beyond WGSL's builtins: the corpus's u64
// words, a strong CAS over WGSL's weak one, 64-bit arithmetic, and libm's
// answers where WGSL's builtins give none (pow of a negative base, atan2 at
// the origin). A u64 load reads the mirror, which an atomic's change never
// reaches: the runtime reads such a word by its atomic, or reads its bits
// that no atomic changes (a cell's target, a task's slot), but a clone of
// an array reads what array atomics changed, so a program with them reads
// the corpus.
const HELPERS = (mirror: boolean) => String.raw`
var<private> NZ: u32;

fn ld64(i: u32) -> vec2<u32> {
  return ${mirror ? "vec2<u32>(P[i], P[i + 1u])"
    : "vec2<u32>(atomicLoad(&M[i]), atomicLoad(&M[i + 1u]))"};
}

fn st32(i: u32, v: u32) {
  atomicStore(&M[i], v);
  P[i] = v;
}

fn st64(i: u32, v: vec2<u32>) {
  st32(i, v.x);
  st32(i + 1u, v.y);
}

fn c_cas(i: u32, x: u32, v: u32) -> u32 {
  loop {
    let r = atomicCompareExchangeWeak(&M[i], x, v);
    if (r.exchanged || r.old_value != x) {
      return r.old_value;
    }
  }
}

fn i2l(x: i32) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(x), select(0u, 0xFFFFFFFFu, x < 0i));
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

fn mulhi(a: u32, b: u32) -> u32 {
  let a0 = a & 0xFFFFu;
  let a1 = a >> 16u;
  let b0 = b & 0xFFFFu;
  let b1 = b >> 16u;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let mid = ((a0 * b0) >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
  return a1 * b1 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
}

fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x * b.x, mulhi(a.x, b.x) + a.x * b.y + a.y * b.x);
}

fn u64_shl(a: vec2<u32>, n: u32) -> vec2<u32> {
  let s = n & 63u;
  if (s == 0u) {
    return a;
  }
  if (s >= 32u) {
    return vec2<u32>(0u, a.x << (s - 32u));
  }
  return vec2<u32>(a.x << s, (a.y << s) | (a.x >> (32u - s)));
}

fn u64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
  let s = n & 63u;
  if (s == 0u) {
    return a;
  }
  if (s >= 32u) {
    return vec2<u32>(a.y >> (s - 32u), 0u);
  }
  return vec2<u32>((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}

fn i64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
  let s = n & 63u;
  let hi = bitcast<i32>(a.y);
  if (s == 0u) {
    return a;
  }
  if (s >= 32u) {
    return vec2<u32>(bitcast<u32>(hi >> (s - 32u)), bitcast<u32>(hi >> 31u));
  }
  return vec2<u32>((a.x >> s) | (a.y << (32u - s)), bitcast<u32>(hi >> s));
}

fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
  return a.y < b.y || (a.y == b.y && a.x < b.x);
}

fn i64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
  let x = bitcast<i32>(a.y);
  let y = bitcast<i32>(b.y);
  return x < y || (x == y && a.x < b.x);
}

// u32 division through i32's, exact for every u32 (a / 0 is a, a % 0 is
// 0, as WGSL has them): Chrome's Metal lane wraps each u32 division in a
// read of a volatile zero (metal_fix_u32_div_mod, crbug.com/517225032), and
// with it the low word a lane returned read 0 across whole SIMD groups of
// Bendcraft's frame, where Dawn without the workaround was right.
fn u32_div(a: u32, b: u32) -> u32 {
  let d = select(b, 1u, b == 0u);
  let q = u32(i32(a >> 1u) / i32(d & 0x7FFFFFFFu)) << 1u;
  return select(q + select(0u, 1u, a - q * d >= d), select(0u, 1u, a >= d),
    d >= 0x80000000u);
}

fn u32_rem(a: u32, b: u32) -> u32 {
  return select(a - u32_div(a, b) * b, 0u, b == 0u);
}

// Long division: q and r of a / b, a word pair when both fit one word.
fn u64_divmod(a: vec2<u32>, b: vec2<u32>) -> array<vec2<u32>, 2> {
  if ((a.y | b.y) == 0u) {
    return array<vec2<u32>, 2>(vec2<u32>(u32_div(a.x, b.x), 0u),
      vec2<u32>(u32_rem(a.x, b.x), 0u));
  }
  var q = vec2<u32>(0u);
  var r = vec2<u32>(0u);
  for (var i = 63i; i >= 0i; i -= 1i) {
    r = u64_shl(r, 1u) | vec2<u32>(u64_shr(a, u32(i)).x & 1u, 0u);
    if (!u64_lt(r, b)) {
      r = u64_sub(r, b);
      q = q | u64_shl(vec2<u32>(1u, 0u), u32(i));
    }
  }
  return array<vec2<u32>, 2>(q, r);
}

fn u64_div(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return u64_divmod(a, b)[0];
}

fn u64_rem(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return u64_divmod(a, b)[1];
}

fn i64_abs(a: vec2<u32>) -> vec2<u32> {
  return select(a, u64_sub(vec2<u32>(0u), a), bitcast<i32>(a.y) < 0i);
}

fn i64_div(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let q = u64_div(i64_abs(a), i64_abs(b));
  return select(q, u64_sub(vec2<u32>(0u), q), ((a.y ^ b.y) >> 31u) != 0u);
}

fn i64_rem(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let r = u64_rem(i64_abs(a), i64_abs(b));
  return select(r, u64_sub(vec2<u32>(0u), r), (a.y >> 31u) != 0u);
}

fn c_log10(x: f32) -> f32 {
  return log2(x) * 0.30102999566398120;
}

fn c_fmod(x: f32, y: f32) -> f32 {
  return x - y * trunc(x / y);
}

fn c_atan2(y: f32, x: f32) -> f32 {
  let pi = select(0.0, 3.14159265358979, (bitcast<u32>(x) >> 31u) != 0u);
  let s = bitcast<f32>((bitcast<u32>(pi) & 0x7FFFFFFFu)
    | (bitcast<u32>(y) & 0x80000000u));
  return select(atan2(y, x), s, y == 0.0 && x == x);
}

fn c_pow(x: f32, y: f32) -> f32 {
  if (y == 0.0) {
    return 1.0;
  }
  if (x >= 0.0 || x != x) {
    return pow(x, y);
  }
  let odd = trunc(y) == y && c_fmod(y, 2.0) != 0.0;
  let m = pow(-x, y);
  return select(select(bitcast<f32>(0x7FC00000u | NZ), m, trunc(y) == y), -m,
    odd);
}
`;

// Kernels
// =======

// The corpus, its mirror and the tables for both, sized (the page puts in
// BEND_M_LEN, the corpus's words) so that no runtime length is divided out;
// the plan writes the run's indirect arguments, a buffer of its own since a
// dispatch cannot read one it binds for writing.
const KERNELS = (tab: number) => String.raw`
@group(0) @binding(0) var<storage, read_write> M:
  array<atomic<u32>, BEND_M_LEN>;
@group(0) @binding(1) var<storage, read> TAB: array<u32, ${tab}>;
@group(0) @binding(2) var<storage, read_write> P: array<u32, BEND_M_LEN>;
@group(1) @binding(0) var<storage, read_write> A: array<u32, 6>;

// The next round's groups, for run's tasks or pack's jobs.
@compute @workgroup_size(1)
fn plan() {
  let n = c_wg_plan(0u);
  let k = c_wg_packing(0u) != 0u;
  A[0] = select(n, 0u, k);
  A[1] = 1u;
  A[2] = 1u;
  A[3] = select(0u, n, k);
  A[4] = 1u;
  A[5] = 1u;
}

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  c_wg_run(0u, g.x);
}

@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) g: vec3<u32>) {
  c_wg_packs(0u, g.x);
}

@compute @workgroup_size(64)
fn window(@builtin(global_invocation_id) g: vec3<u32>) {
  c_wg_window(0u, g.x);
}
`;

// Glue
// ====

// The host's side, included by the template's Gpu section in a page build
// (BEND_WEBGPU). The device's corpus is a buffer of its own, laid out as
// the host's, so a bang copies its task and arguments in, as an image its
// heap starts with, runs rounds until they stop, and copies the result out
// of R, where the rounds packed it, into the host's heap (a result of words
// alone, a Unit or a number, reads nothing back). An Image stays on the
// device when the host never opens one (HOST_IMAGE), as native Metal's
// shared heap keeps it: Window.frame draws it there, a drop uncounts it,
// and a later bang's copy takes it from the device's words. A copy walks a
// node at a time off a stack of (term, slot) jobs and keeps a shared cell
// once, its count the references it met. WebGPU lives on the page's main
// thread: the program posts a request there and waits on its first word,
// 1 when done and 2 when WebGPU failed (the console says why).
const GLUE = String.raw`
#include <emscripten.h>
#include <emscripten/threading.h>
${WG_DEFS}
#define GPU_IMG  (HEAP_OFF + PAGE_LEN)
#define GPU_HEAD (H_BANK + 3 * NCLS_ALL)

typedef struct { u32 at; u32 ptr; u32 words; } GpuPut;

typedef struct {
  u32    done;
  GpuPut put[4];
  u32    zero_at;
  u32    zero_words;
  u32    stop;
  u32    back;
  u32    back_words;
  u32    fid;
  u32    ord;
} GpuReq;

typedef struct {
  const u64* src;
  u64        src_at;
  u64        src_len;
  u64*       dst;
  u64        dst_at;
  u64        top;
  u64        cap;
  Env        e;
  u64*       job;
  u32        jobs;
  u32        job_cap;
  u64*       key;
  u64*       val;
  u32        keys;
  u32        key_cap;
  bool       keep;
} Seam;

static u32    gpu_words;
static GpuReq gpu_req;
static Seam   gpu_seam;
static u64    gpu_head[GPU_HEAD];
static u64*   gpu_arena;
static u64    gpu_arena_len;
static u64    gpu_live;

#define gpu_span()  (1ull << 30)
#define gpu_make(p) true
#define gpu_load(b)

EM_JS(void, webgpu_js_open, (const char* src, const u32* tab, u32 n,
  GpuReq* q, u32 win, u32 pix), {
  var end = function(v, why) {
    if (why) {
      Module.bendOn = "the ! on the cores: " + why;
      console.warn("bend: " + Module.bendOn);
    }
    Atomics.store(HEAP32, q >> 2, v);
    Atomics.notify(HEAP32, q >> 2);
  };
  (async function() {
    // A laptop with two GPUs hands out its integrated one by default, the
    // one that also draws the screen; a software adapter (SwiftShader, let
    // through by a flag) runs the rounds on the CPU, slower than the cores.
    var ad = navigator.gpu && await navigator.gpu.requestAdapter({
      powerPreference: "high-performance" });
    if (!ad || ad.info.isFallbackAdapter) {
      return end(2, "no hardware WebGPU adapter");
    }
    var L = ad.limits;
    var dev = await ad.requestDevice({ requiredLimits: {
      maxBufferSize: L.maxBufferSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize } });
    var G = { dev: dev, bad: null, ks: {}, ords: {} };
    dev.addEventListener("uncapturederror", function(ev) {
      G.bad = G.bad || ev.error.message;
      console.error("bend: WebGPU: " + ev.error.message);
    });
    dev.lost.then(function(info) { G.bad = G.bad || "device lost: " + info.message; });
    var U = GPUBufferUsage;
    var bytes = Math.floor(Math.min(L.maxBufferSize,
      L.maxStorageBufferBindingSize, 2 ** 31) / 65536) * 65536;
    for (;;) {
      dev.pushErrorScope("out-of-memory");
      G.M = dev.createBuffer({ size: bytes,
        usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
      G.P = dev.createBuffer({ size: bytes, usage: U.STORAGE | U.COPY_DST });
      if (!await dev.popErrorScope()) {
        break;
      }
      G.M.destroy();
      G.P.destroy();
      if (bytes <= 2 ** 29) {
        return end(2, "no room for a corpus and its mirror");
      }
      bytes = Math.floor(bytes / 2 / 65536) * 65536;
    }
    // WebGPU zeroes a buffer at its first use: done here, while the shader
    // compiles, since the first ! would otherwise pay ~100 ms for the
    // corpora (Apple M5), whatever its size.
    var enc = dev.createCommandEncoder();
    enc.clearBuffer(G.M);
    enc.clearBuffer(G.P);
    dev.queue.submit([enc.finish()]);
    G.T = dev.createBuffer({ size: Math.max(n, 1) * 4,
      usage: U.STORAGE | U.COPY_DST });
    dev.queue.writeBuffer(G.T, 0, HEAPU32.slice(tab >> 2, (tab >> 2) + n));
    G.A = dev.createBuffer({ size: 32, usage: U.STORAGE | U.INDIRECT });
    var mod = dev.createShaderModule({ code: UTF8ToString(src)
      .replace(/BEND_M_LEN/g, bytes / 4 + "u") });
    var bad = (await mod.getCompilationInfo()).messages.filter(function(m) {
      return m.type === "error";
    });
    if (bad.length > 0) {
      return end(2, "the WGSL fails at " + bad[0].lineNum + ": "
        + bad[0].message);
    }
    var C = GPUShaderStage.COMPUTE;
    var b0 = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: C, buffer: { type: "storage" } },
      { binding: 1, visibility: C, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: C, buffer: { type: "storage" } }] });
    var b1 = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: C, buffer: { type: "storage" } }] });
    var pipe = function(name, bs) {
      return dev.createComputePipelineAsync({ compute: { module: mod,
        entryPoint: name }, layout: dev.createPipelineLayout({
        bindGroupLayouts: bs }) });
    };
    G.plan = await pipe("plan", [b0, b1]);
    G.run  = await pipe("run", [b0]);
    G.pack = await pipe("pack", [b0]);
    G.win  = await pipe("window", [b0]);
    G.g0 = dev.createBindGroup({ layout: b0, entries: [
      { binding: 0, resource: { buffer: G.M } },
      { binding: 1, resource: { buffer: G.T } },
      { binding: 2, resource: { buffer: G.P } }] });
    G.g1 = dev.createBindGroup({ layout: b1, entries: [
      { binding: 0, resource: { buffer: G.A } }] });
    // A band of a kept Image's square drawn into the third queue
    // (wg_window): its node, then its size and level, in four words.
    G.band = function(lo, hi, a, b) {
      var args = new Uint32Array([lo, hi, a, b]);
      dev.queue.writeBuffer(G.M, win * 8, args);
      dev.queue.writeBuffer(G.P, win * 8, args);
      var enc = dev.createCommandEncoder();
      var pass = enc.beginComputePass();
      pass.setPipeline(G.win);
      pass.setBindGroup(0, G.g0);
      pass.dispatchWorkgroups(Math.ceil((a & 0xFFFF) * (a >>> 16) / 64));
      pass.end();
      return enc;
    };
    G.pix = pix * 8;
    Module.bendOn = "the ! on " + [ad.info.vendor, ad.info.architecture,
      ad.info.description].filter(Boolean).join(" ");
    await dev.queue.onSubmittedWorkDone();
    Module.bendGpu = G;
    HEAPU32[(q >> 2) + 3] = bytes / 8;
    end(1);
  })().catch(function(e) { end(2, String(e)); });
});

// The rounds, as many a submit as the last bang of the same function
// planned (the count after the stop word; 16 at first) and doubling to 256, the header read back
// after each submit until its stop word is set. A wait on the header costs
// about a millisecond, and so do some twenty rounds past the stop. The
// first four bangs of a function alternate the kids' order (WG_ORD), from
// a slot a kid, and the later ones keep the order of the fastest.
EM_JS(void, webgpu_js_run, (GpuReq* q), {
  var G = Module.bendGpu;
  var w = function(k) { return HEAPU32[(q >> 2) + k]; };
  var end = function(v) {
    Atomics.store(HEAP32, q >> 2, v);
    Atomics.notify(HEAP32, q >> 2);
  };
  (async function() {
    var dq = G.dev.queue;
    var bw = w(17) * 8;
    var enc = G.dev.createCommandEncoder();
    for (var i = 0; i < 4; i += 1) {
      var p = w(2 + 3 * i);
      var at = w(1 + 3 * i) * 8;
      var n = w(3 + 3 * i) * 8;
      if (n > 0) {
        dq.writeBuffer(G.M, at, HEAPU8.slice(p, p + n));
        enc.copyBufferToBuffer(G.M, at, G.P, at, n);
      }
    }
    var o = G.ords[w(18)] || (G.ords[w(18)] = { n: 0, t: [Infinity, Infinity] });
    var ord = o.n < 4 ? o.n & 1 : Number(o.t[1] < o.t[0]);
    dq.writeBuffer(G.M, w(19) * 8, new Uint32Array([ord]));
    var t0 = performance.now();
    if (!G.head || G.head.size < bw) {
      G.head = G.dev.createBuffer({ size: bw,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    enc.clearBuffer(G.M, w(13) * 8, w(14) * 8);
    enc.clearBuffer(G.P, w(13) * 8, w(14) * 8);
    for (var k = G.ks[w(18)] || 16; !G.bad; k = Math.min(k * 2, 256)) {
      var pass = enc.beginComputePass();
      for (var r = 0; r < k; r += 1) {
        pass.setPipeline(G.plan);
        pass.setBindGroup(0, G.g0);
        pass.setBindGroup(1, G.g1);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(G.run);
        pass.setBindGroup(0, G.g0);
        pass.dispatchWorkgroupsIndirect(G.A, 0);
        pass.setPipeline(G.pack);
        pass.setBindGroup(0, G.g0);
        pass.dispatchWorkgroupsIndirect(G.A, 12);
      }
      pass.end();
      enc.copyBufferToBuffer(G.M, 0, G.head, 0, bw);
      dq.submit([enc.finish()]);
      await G.head.mapAsync(GPUMapMode.READ, 0, bw);
      var h = new Uint32Array(G.head.getMappedRange(0, bw));
      HEAPU32.set(h, w(16) >> 2);
      var stop = h[2 * w(15)] !== 0;
      var used = h[2 * w(15) + 2] + 1;
      G.head.unmap();
      if (stop) {
        G.ks[w(18)] = Math.min(used, 256);
        o.t[ord] = Math.min(o.t[ord], performance.now() - t0);
        o.n += 1;
        return end(1);
      }
      enc = G.dev.createCommandEncoder();
    }
    end(2);
  })().catch(function(e) {
    console.error("bend: WebGPU: " + e);
    end(2);
  });
});

EM_JS(void, webgpu_js_read, (GpuReq* q), {
  var G = Module.bendGpu;
  var at = HEAPU32[(q >> 2) + 1] * 8;
  var to = HEAPU32[(q >> 2) + 2];
  var n = HEAPU32[(q >> 2) + 3] * 8;
  var end = function(v) {
    Atomics.store(HEAP32, q >> 2, v);
    Atomics.notify(HEAP32, q >> 2);
  };
  (async function() {
    if (!G.rb || G.rb.size < n) {
      if (G.rb) {
        G.rb.destroy();
      }
      G.rb = G.dev.createBuffer({ size: Math.max(n, 1 << 20),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    var enc = G.dev.createCommandEncoder();
    enc.copyBufferToBuffer(G.M, at, G.rb, 0, n);
    G.dev.queue.submit([enc.finish()]);
    await G.rb.mapAsync(GPUMapMode.READ, 0, n);
    HEAPU8.set(new Uint8Array(G.rb.getMappedRange(0, n)), to);
    G.rb.unmap();
    end(G.bad ? 2 : 1);
  })().catch(function(e) {
    console.error("bend: WebGPU: " + e);
    end(2);
  });
});

// A band of a kept Image's square drawn where it lives and its rows read
// back into the frame, stride words apart.
EM_JS(void, webgpu_js_window, (GpuReq* q, u32 lo, u32 hi, u32 a, u32 b,
  u32* out, u32 stride), {
  var G = Module.bendGpu;
  var sw = (a & 0xFFFF) * 4;
  var pw = ((a & 0xFFFF) + 63 & ~63) * 4;
  var n = pw * (a >>> 16);
  var end = function(v) {
    Atomics.store(HEAP32, q >> 2, v);
    Atomics.notify(HEAP32, q >> 2);
  };
  (async function() {
    if (!G.rb || G.rb.size < n) {
      if (G.rb) {
        G.rb.destroy();
      }
      G.rb = G.dev.createBuffer({ size: Math.max(n, 1 << 20),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    var enc = G.band(lo, hi, a, b);
    enc.copyBufferToBuffer(G.M, G.pix, G.rb, 0, n);
    G.dev.queue.submit([enc.finish()]);
    await G.rb.mapAsync(GPUMapMode.READ, 0, n);
    var m = new Uint8Array(G.rb.getMappedRange(0, n));
    for (var j = 0; j < n; j += pw) {
      HEAPU8.set(m.subarray(j, j + sw), out + j / pw * stride * 4);
    }
    G.rb.unmap();
    end(G.bad ? 2 : 1);
  })().catch(function(e) {
    console.error("bend: WebGPU: " + e);
    end(2);
  });
});

static bool gpu_wait(GpuReq* q) {
  while (a32_load_acq(&q->done) == 0) {
    emscripten_futex_wait(&q->done, 0, 1000);
  }
  return q->done == 1;
}

static void gpu_done(GpuReq* q) {
  if (!gpu_wait(q)) {
    err_fail("the GPU failed (WebGPU; the console says why)");
  }
}

static void gpu_ask(GpuReq* q, bool run) {
  if (run) {
    MAIN_THREAD_ASYNC_EM_ASM({ webgpu_js_run($0); }, q);
  } else {
    MAIN_THREAD_ASYNC_EM_ASM({ webgpu_js_read($0); }, q);
  }
  gpu_done(q);
}

static bool gpu_probe(void) {
  MAIN_THREAD_ASYNC_EM_ASM({ webgpu_js_open($0, $1, $2, $3, $4, $5); },
    GPU_SRC, GPU_TAB, sizeof GPU_TAB / 4, &gpu_req, WG_WIN, (u32)wg_qat(2));
  bool ok = gpu_wait(&gpu_req);
  gpu_words = gpu_req.put[0].words;
  return ok && gpu_words > GPU_IMG + CUBE * PAGE_LEN;
}

static void* seam_grow(void* p, u64 bytes) {
  p = realloc(p, bytes);
  if (p == NULL) {
    err_fail("the GPU seam ran out of memory");
  }
  return p;
}

// The device's words from at on, into the arena.
static void gpu_read(Loc at, u64 len) {
  GpuReq* q = &gpu_req;
  if (len > gpu_arena_len) {
    gpu_arena_len = len;
    gpu_arena     = seam_grow(gpu_arena, len * 8);
  }
  *q = (GpuReq){ 0 };
  q->put[0] = (GpuPut){ (u32)at, (u32)(uintptr_t)gpu_arena, (u32)len };
  gpu_ask(q, false);
}

#define seam_src(s, w) (s)->src[(w) - (s)->src_at]
#define seam_dst(s, w) (s)->dst[(w) - (s)->dst_at]

// A node's room: its class in the host's heap, else the image's next words.
static Loc seam_alloc(Seam* s, Cls c) {
  if (s->top == 0) {
    return heap_alloc(s->e, c);
  }
  Loc l   = s->top;
  s->top += 1ull << c;
  if (s->top - s->dst_at > s->cap) {
    s->cap = 2 * (s->top - s->dst_at);
    s->dst = seam_grow(s->dst, s->cap * 8);
  }
  return l;
}

static void seam_job(Seam* s, Term t, Loc slot) {
  if (term_triv(t)) {
    seam_dst(s, slot) = t;
    return;
  }
  if (s->jobs == s->job_cap) {
    s->job_cap = s->job_cap ? 2 * s->job_cap : 1024;
    s->job     = seam_grow(s->job, s->job_cap * 16ull);
  }
  s->job[2 * s->jobs]     = t;
  s->job[2 * s->jobs + 1] = slot;
  s->jobs += 1;
}

// An Image the copy out keeps where it is (keep, WG_KEEP): the host holds
// its term with the device's node past GPU_FAR.
static bool seam_kept(Seam* s, Term t) {
  return s->keep && term_tag(t) == TAG_CTR && term_aux(t) == WG_IMAGE;
}

static Loc seam_keep(Loc at) {
  gpu_owed += 1;
  return GPU_FAR | at;
}

static Loc seam_far(Seam* s, Term t, Loc at);

static Loc seam_node(Seam* s, Term t, Loc at) {
  u64  tag = term_tag(t);
  u32  aux = (u32)term_aux(t);
  bool blk = tag == TAG_ARR || tag == TAG_BUF;
  u32  n   = tag == TAG_CTR ? cid_arity(aux) : tag == TAG_CLO
    ? fid_arity(aux) - 1 : 1u << blk_span(t);
  if ((!blk && tag != TAG_CTR && tag != TAG_CLO) || at < s->src_at
    || at + n > s->src_at + s->src_len) {
    err_fail("a term the GPU seam cannot copy");
  }
  Loc l = seam_alloc(s, blk ? blk_span(t) : cls_fit(n));
  if (tag == TAG_BUF) {
    memcpy(&seam_dst(s, l), &seam_src(s, at), n * 8ull);
  } else {
    for (u32 j = 0; j < n; j += 1) {
      seam_job(s, seam_src(s, at + j), l + j);
    }
  }
  return l;
}

// A node's copy: a kept Image's from the device's words, one the copy out
// keeps as its node there, else the node itself.
static Loc seam_copy(Seam* s, Term t, Loc at) {
  return at >= GPU_FAR ? seam_far(s, t, at & ~GPU_FAR)
    : seam_kept(s, t) ? seam_keep(at) : seam_node(s, t, at);
}

// A shared cell: copied at its first reference, counted at the others.
static Loc seam_cell(Seam* s, Term t) {
  Loc r = term_loc(t);
  if (2 * (s->keys + 1) > s->key_cap) {
    u64* ks = s->key;
    u64* vs = s->val;
    u32  kn = s->key_cap;
    s->key_cap = kn ? 2 * kn : 1024;
    s->key     = calloc(s->key_cap, 8);
    s->val     = seam_grow(NULL, s->key_cap * 8ull);
    if (s->key == NULL) {
      err_fail("the GPU seam ran out of memory");
    }
    for (u32 i = 0; i < kn; i += 1) {
      u32 h = (u32)(ks[i] * 0x9E3779B97F4A7C15ull >> 40) & (s->key_cap - 1);
      while (ks[i] != 0 && s->key[h] != 0) {
        h = (h + 1) & (s->key_cap - 1);
      }
      if (ks[i] != 0) {
        s->key[h] = ks[i];
        s->val[h] = vs[i];
      }
    }
    free(ks);
    free(vs);
  }
  u32 h = (u32)(r * 0x9E3779B97F4A7C15ull >> 40) & (s->key_cap - 1);
  while (s->key[h] != 0 && s->key[h] != r) {
    h = (h + 1) & (s->key_cap - 1);
  }
  if (s->key[h] == r) {
    if ((seam_dst(s, s->val[h]) & RFC_CNT) == RFC_CNT) {
      err_post(s->e.mem, ERR_RFCS);
    }
    seam_dst(s, s->val[h]) += 1;
    return s->val[h];
  }
  if (r < s->src_at || r >= s->src_at + s->src_len) {
    err_fail("a term the GPU seam cannot copy");
  }
  s->keys  += 1;
  s->key[h] = r;
  s->val[h] = seam_alloc(s, 0);
  Loc nt = seam_copy(s, t, seam_src(s, r) >> 24);
  seam_dst(s, s->val[h]) = ((u64)nt << 24) | 1;
  return s->val[h];
}

static void seam_run(Seam* s) {
  while (s->jobs > 0) {
    s->jobs -= 1;
    Term t    = s->job[2 * s->jobs];
    Loc  slot = s->job[2 * s->jobs + 1];
    Loc  l    = term_rfc(t) ? seam_cell(s, t) : seam_copy(s, t, term_loc(t));
    seam_dst(s, slot) = (t & ~LOC_MASK) | l;
  }
  if (s->keys > 0) {
    memset(s->key, 0, s->key_cap * 8ull);
    s->keys = 0;
  }
}

// A kept Image in a !'s argument: the device's live words come back and
// its tree is copied from them into the argument's image, the host's term
// left as it was.
static Loc seam_far(Seam* s, Term t, Loc at) {
  static Seam f;
  gpu_read(GPU_IMG, gpu_live - GPU_IMG);
  f = (Seam){ gpu_arena, GPU_IMG, gpu_live - GPU_IMG, s->dst, s->dst_at,
    s->top, s->cap, s->e, f.job, 0, f.job_cap, f.key, f.val, 0, f.key_cap,
    false };
  Loc l = seam_node(&f, t, at);
  seam_run(&f);
  s->dst = f.dst;
  s->top = f.top;
  s->cap = f.cap;
  return l;
}

// A bang's task and the arguments its def owns, dropped where they are:
// the device has copies. The boxes it borrows (BANG_BRW) stay the
// caller's, which drops them again later.
static void seam_sink(Env e, Fid fid, Loc a, u32 ar) {
  for (u32 j = 0; j < ar; j += 1) {
    bool brw = false;
    for (u32 i = 0; i < BANG_BRWS; i += 1) {
      brw = brw || (BANG_BRW[2 * i] == fid && BANG_BRW[2 * i + 1] == j);
    }
    if (!brw) {
      term_sink(e, e.mem[a + j]);
    }
  }
  heap_free(e, cls_fit(ar + 2), a);
}

// A kept Image's term on the device, else 0.
static Term gpu_root(Term t) {
  Loc l = term_rfc(t) ? CORPUS[term_loc(t)] >> 24 : term_loc(t);
  return term_tag(t) == TAG_CTR && l >= GPU_FAR
    ? (t & ~(RFC_BIT | LOC_MASK)) | (l & ~GPU_FAR) : 0;
}

// A kept Image's square at (x, y) in an Image the host built, of side 2^i
// and clipped to the w x h frame, drawn where it lives, a band of rows a
// dispatch: only its pixels come back (window_host_at's, on a page).
static void gpu_window(Term t, u32 i, u32 x, u32 y, u32 w, u32 h, u32* out) {
  Term n    = gpu_root(t);
  u32  sw   = (1u << i) < w - x ? 1u << i : w - x;
  u32  sh   = (1u << i) < h - y ? 1u << i : h - y;
  u32  band = 65535 * 64 / ((sw + 63) & ~63u);
  for (u32 r = 0; r < sh; r += band) {
    u32 rows = band < sh - r ? band : sh - r;
    gpu_req = (GpuReq){ 0 };
    MAIN_THREAD_ASYNC_EM_ASM({ webgpu_js_window($0, $1, $2, $3, $4, $5,
      $6); }, &gpu_req, (u32)n, (u32)(n >> 32), sw | rows << 16, i | r << 8,
      out + (u64)(y + r) * w + x, w);
    gpu_done(&gpu_req);
  }
}

// The bang cube_run hands the GPU: its task alone on the host's ring 0. A
// def that returns Unit gives Unit{} (fid_unit), so its bang needs no
// device.
static void gpu_pass(u32 f) {
  Corpus H   = CORPUS;
  Env    e   = { H, ALC[0] };
  u32*   get = ring_get(H, 0);
  Term   t   = *ring_slot(H, 0, *get) & ~RFC_BIT;
  Seam*  s   = &gpu_seam;
  *get += 1;
  Fid    fid = (u32)term_aux(t);
  u32    ar  = fid_arity(fid);
  Loc    a   = term_loc(t);
  if (fid_unit(fid)) {
    seam_sink(e, fid, a, ar);
    H[H_ROOT_WORD] = term_pak(CID_UNIT, 0);
    a32_store_rel(a32_at(H, H_ROOT_DONE), 2);
    return;
  }
  // One generation of kept Images at a time: while the host holds one,
  // this bang's image starts past them and it keeps nothing of its own.
  bool   keep = gpu_owed == 0;
  s->src     = H;
  s->src_at  = 0;
  s->src_len = ~0ull >> 1;
  s->dst_at  = keep ? GPU_IMG : gpu_live;
  s->top     = s->dst_at;
  s->e       = e;
  Loc tl = seam_alloc(s, cls_fit(ar + 2));
  for (u32 j = 0; j < ar; j += 1) {
    seam_job(s, H[a + j], tl + j);
  }
  seam_run(s);
  seam_dst(s, tl + ar)     = TERM_HOLE;
  seam_dst(s, tl + ar + 1) = 0;
  seam_sink(e, fid, a, ar);
  static u64 task;
  task = term_tsk(fid, tl);
  memset(gpu_head, 0, sizeof gpu_head);
  gpu_head[H_BUMP]  = (s->top - HEAP_OFF + PAGE_LEN - 1) >> PAGE_BITS;
  gpu_head[H_CAP]   = (gpu_words - HEAP_OFF) >> PAGE_BITS;
  gpu_head[WG_NOUT] = 1;
  gpu_head[WG_KEEP] = keep;
  if (gpu_head[H_BUMP] >= gpu_head[H_CAP]) {
    err_post(H, ERR_HEAP);
  }
  GpuReq* q = &gpu_req;
  *q = (GpuReq){ 0 };
  q->put[0] = (GpuPut){ 0, (u32)(uintptr_t)gpu_head, GPU_HEAD };
  q->put[1] = (GpuPut){ STAT_OFF, (u32)(uintptr_t)(H + STAT_OFF), STAT_LEN };
  q->put[2] = (GpuPut){ (u32)s->dst_at, (u32)(uintptr_t)s->dst,
    (u32)(s->top - s->dst_at) };
  q->put[3] = (GpuPut){ (u32)wg_qat(1), (u32)(uintptr_t)&task, 1 };
  q->zero_at    = ALC_OFF;
  q->zero_words = CUBE * 2 * ALC_WORDS;
  q->stop       = WG_STOP;
  q->back       = (u32)(uintptr_t)gpu_head;
  q->back_words = GPU_HEAD;
  q->fid        = fid;
  q->ord        = WG_ORD;
  gpu_ask(q, true);
  if ((u32)gpu_head[H_ERROR_CODE] != 0) {
    err_post(H, (u32)gpu_head[H_ERROR_CODE]);
  }
  u32 done = (u32)gpu_head[H_ROOT_DONE];
  if (done == 0) {
    err_fail("frontier drained without a result");
  }
  u64 len = (u32)gpu_head[WG_PTOP];
  Loc r0  = (u32)gpu_head[WG_PR0];
  if (len > 0) {
    gpu_read(r0, len);
  }
  Seam out = { gpu_arena, r0, len, H, 0, 0, 0, e, s->job, 0,
    s->job_cap, s->key, s->val, 0, s->key_cap, keep };
  for (u32 j = 0; j + 1 < done; j += 1) {
    seam_job(&out, gpu_head[H_ROOT_WORD + j], H_ROOT_WORD + j);
  }
  seam_run(&out);
  s->job     = out.job;
  s->job_cap = out.job_cap;
  s->key     = out.key;
  s->val     = out.val;
  s->key_cap = out.key_cap;
  if (keep) {
    gpu_live = r0;
  }
  a32_store_rel(a32_at(H, H_ROOT_DONE), done);
}
`;

// Export
// ======

// The page's lane for a program with `!`: the glue, with the WGSL and
// TAB's words it hands the browser, written into dir, and its path, for
// the template's BEND_WEBGPU.
export function webgpu_page(c: string, dir: string, cc = "clang"): string {
  const dev = device_of(ast_of(c, dir, cc));
  const tab = dev.tab.length > 0 ? dev.tab : [0];
  const mirror = !/\ba32_\w+\(blk_ptr\(/.test(c);
  const src = (KERNELS(tab.length) + HELPERS(mirror) + "\n" + dev.code)
    .split("\n");
  const glue = path.join(dir, "webgpu.c");
  fs.writeFileSync(glue, "static const char GPU_SRC[] =\n"
    + src.map((l) => JSON.stringify(l + "\n")).join("\n") + ";\n\n"
    + "static const u32 GPU_TAB[] = { " + tab.join(", ") + " };\n" + GLUE);
  return glue;
}
