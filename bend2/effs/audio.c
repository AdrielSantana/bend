// Audio
// =====

// The ring: 4096 float32 stereo frames. The effects fill it on the
// evaluator's thread; the device's callback drains it and pads the
// rest of its buffer with silence. All audio effects share this source;
// each entry is present only when its effect is reachable.
#ifndef IO_RING
#define IO_RING 4096u

#ifdef __OBJC__
#import <AudioToolbox/AudioToolbox.h>
#elif defined(__linux__)
#include <alsa/asoundlib.h>
#elif defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/threading.h>
#endif

typedef struct {
  _Atomic(u64) read, written;
  float        pcm[IO_RING * 2];
#ifdef __OBJC__
  AudioUnit    unit;
#elif defined(__linux__)
  snd_pcm_t*   unit;
  pthread_t    pump;
  _Atomic(u32) done;
#endif
} IoRing;

// n frames queued after the write; a write past the ring's room is
// dropped and the queue answered as it is.
static u64 io_ring_write(IoRing* p, const float* pcm, u32 n) {
  u64 w = atomic_load_explicit(&p->written, memory_order_relaxed);
  u64 r = atomic_load_explicit(&p->read, memory_order_acquire);
  if (w - r + n <= IO_RING) {
    u32 at    = (u32)(w % IO_RING);
    u32 first = n < IO_RING - at ? n : IO_RING - at;
    memcpy(p->pcm + at * 2, pcm, first * 8);
    memcpy(p->pcm, pcm + first * 2, (n - first) * 8);
    atomic_store_explicit(&p->written, w + n, memory_order_release);
    w += n;
  }
  return w - r;
}

static void io_ring_pull(IoRing* p, float* out, u32 frames) {
  u64 r     = atomic_load_explicit(&p->read, memory_order_relaxed);
  u64 w     = atomic_load_explicit(&p->written, memory_order_acquire);
  u32 n     = (u32)(w - r < frames ? w - r : frames);
  u32 at    = (u32)(r % IO_RING);
  u32 first = n < IO_RING - at ? n : IO_RING - at;
  memcpy(out, p->pcm + at * 2, first * 8);
  memcpy(out + first * 2, p->pcm, (n - first) * 8);
  memset(out + n * 2, 0, (frames - n) * 8);
  atomic_store_explicit(&p->read, r + n, memory_order_release);
}

#ifdef __OBJC__

static OSStatus io_ring_pump(void* ctx, AudioUnitRenderActionFlags* flags,
  const AudioTimeStamp* when, UInt32 bus, UInt32 frames,
  AudioBufferList* bl) {
  if (bl->mNumberBuffers != 1 || bl->mBuffers[0].mNumberChannels != 2
      || bl->mBuffers[0].mDataByteSize < frames * 8) {
    return kAudio_ParamError;
  }
  io_ring_pull(ctx, bl->mBuffers[0].mData, frames);
  return noErr;
}

// The default output unit fed float32 stereo at the asked rate (the
// unit converts to the device's own).
static u32 io_ring_start(IoRing* p, u32 rate) {
  AudioComponentDescription desc = { kAudioUnitType_Output,
    kAudioUnitSubType_DefaultOutput, kAudioUnitManufacturer_Apple, 0, 0 };
  AudioComponent comp = AudioComponentFindNext(NULL, &desc);
  if (comp == NULL || AudioComponentInstanceNew(comp, &p->unit) != noErr) {
    return ENODEV;
  }
  AudioStreamBasicDescription fmt = { 0 };
  fmt.mSampleRate       = rate;
  fmt.mFormatID         = kAudioFormatLinearPCM;
  fmt.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
  fmt.mBytesPerPacket   = 8;
  fmt.mFramesPerPacket  = 1;
  fmt.mBytesPerFrame    = 8;
  fmt.mChannelsPerFrame = 2;
  fmt.mBitsPerChannel   = 32;
  AURenderCallbackStruct cb = { io_ring_pump, p };
  bool ok = AudioUnitSetProperty(p->unit, kAudioUnitProperty_StreamFormat,
      kAudioUnitScope_Input, 0, &fmt, sizeof fmt) == noErr
    && AudioUnitSetProperty(p->unit, kAudioUnitProperty_SetRenderCallback,
      kAudioUnitScope_Input, 0, &cb, sizeof cb) == noErr
    && AudioUnitInitialize(p->unit) == noErr
    && AudioOutputUnitStart(p->unit) == noErr;
  return ok ? 0 : ENODEV;
}

static void io_ring_free(IoRing* p) {
  if (p->unit != NULL) {
    AudioOutputUnitStop(p->unit);
    AudioUnitUninitialize(p->unit);
    AudioComponentInstanceDispose(p->unit);
  }
  free(p);
}

#elif defined(__linux__)

// A thread feeds the default ALSA device 256 frames at a time (a
// write blocks until the device has room, so the ring drains at the
// device's clock).
static void* io_ring_pump(void* ctx) {
  IoRing* p = ctx;
  float   out[256 * 2];
  while (atomic_load_explicit(&p->done, memory_order_relaxed) == 0) {
    io_ring_pull(p, out, 256);
    snd_pcm_sframes_t n = snd_pcm_writei(p->unit, out, 256);
    if (n < 0) {
      snd_pcm_recover(p->unit, (int)n, 1);
    }
  }
  return NULL;
}

static void io_ring_hush(const char* file, int line, const char* fn, int err,
  const char* fmt, ...) {
}

static u32 io_ring_start(IoRing* p, u32 rate) {
  snd_lib_error_set_handler(io_ring_hush);
  if (snd_pcm_open(&p->unit, "default", SND_PCM_STREAM_PLAYBACK, 0) < 0) {
    return ENODEV;
  }
  if (snd_pcm_set_params(p->unit, SND_PCM_FORMAT_FLOAT_LE,
    SND_PCM_ACCESS_RW_INTERLEAVED, 2, rate, 1, 20000) < 0
    || pthread_create(&p->pump, NULL, io_ring_pump, p) != 0) {
    return ENODEV;
  }
  return 0;
}

static void io_ring_free(IoRing* p) {
  if (p->unit != NULL) {
    atomic_store_explicit(&p->done, 1, memory_order_relaxed);
    if (p->pump != 0) {
      pthread_join(p->pump, NULL);
    }
    snd_pcm_close(p->unit);
  }
  free(p);
}

#elif defined(__EMSCRIPTEN__)

// An AudioWorklet drains the ring out of the page's shared memory. The
// AudioContext lives on the main thread, so an open and a close run there and this worker waits for
// their answer (1 done, 2 failed); a context starts suspended until the
// page's first key or click.
EM_JS(void, audio_js_open, (IoRing* p, _Atomic(u64)* read, float* pcm,
  u32 frames, u32 rate, _Atomic(u32)* done), {
  var ring = function() {
    registerProcessor("bend-ring", class extends AudioWorkletProcessor {
      constructor(o) {
        super();
        var m = o.processorOptions;
        this.at  = new BigUint64Array(m.memory, m.read, 2);
        this.pcm = new Float32Array(m.memory, m.pcm, m.frames * 2);
      }
      process(ins, outs) {
        var l = outs[0][0];
        var r = outs[0][1];
        var k = this.pcm.length / 2;
        var from = Atomics.load(this.at, 0);
        var n = Math.min(Number(Atomics.load(this.at, 1) - from), l.length);
        var at = Number(from % BigInt(k));
        for (var i = 0; i < n; i += 1) {
          var j = (at + i) % k * 2;
          l[i] = this.pcm[j];
          r[i] = this.pcm[j + 1];
        }
        l.fill(0, n);
        r.fill(0, n);
        Atomics.store(this.at, 0, from + BigInt(n));
        return true;
      }
    });
  };
  var answer = function(v) {
    Atomics.store(HEAP32, done >> 2, v);
    Atomics.notify(HEAP32, done >> 2);
  };
  var ctx, url;
  var go = function() {
    if (ctx.state === "suspended") {
      ctx.resume();
    }
  };
  new Promise(function(ok) {
    ctx = new AudioContext({ sampleRate: rate });
    url = URL.createObjectURL(new Blob(["(" + ring + ")()"],
      { type: "text/javascript" }));
    ok(ctx.audioWorklet.addModule(url));
  }).then(function() {
    var node = new AudioWorkletNode(ctx, "bend-ring", { numberOfInputs: 0,
      outputChannelCount: [2], processorOptions: { memory: HEAPU8.buffer,
      read: read, pcm: pcm, frames: frames } });
    node.connect(ctx.destination);
    (Module.bendAudio = Module.bendAudio || new Map()).set(p, { ctx: ctx, go: go });
    window.addEventListener("keydown", go);
    window.addEventListener("pointerdown", go);
    go();
    answer(1);
  }).catch(function(e) {
    console.warn("bend: Audio.open: " + e);
    if (ctx) {
      ctx.close();
    }
    answer(2);
  }).finally(function() {
    URL.revokeObjectURL(url);
  });
});

EM_JS(void, audio_js_close, (IoRing* p, _Atomic(u32)* done), {
  var a = Module.bendAudio && Module.bendAudio.get(p);
  var answer = function() {
    Atomics.store(HEAP32, done >> 2, 1);
    Atomics.notify(HEAP32, done >> 2);
  };
  if (!a) {
    return answer();
  }
  Module.bendAudio.delete(p);
  window.removeEventListener("keydown", a.go);
  window.removeEventListener("pointerdown", a.go);
  a.ctx.close().then(answer, answer);
});

static u32 io_ring_wait(_Atomic(u32)* done) {
  while (atomic_load_explicit(done, memory_order_acquire) == 0) {
    emscripten_futex_wait((void*)done, 0, 1000);
  }
  return atomic_load_explicit(done, memory_order_acquire);
}

static u32 io_ring_start(IoRing* p, u32 rate) {
  _Atomic(u32) done = 0;
  MAIN_THREAD_ASYNC_EM_ASM({ audio_js_open($0, $1, $2, $3, $4, $5); }, p,
    &p->read, p->pcm, IO_RING, rate, &done);
  return io_ring_wait(&done) == 1 ? 0 : ENODEV;
}

// The ring is freed once its context is closed: the worklet writes its
// read count into it until then.
static void io_ring_free(IoRing* p) {
  _Atomic(u32) done = 0;
  MAIN_THREAD_ASYNC_EM_ASM({ audio_js_close($0, $1); }, p, &done);
  io_ring_wait(&done);
  free(p);
}

#else

static u32 io_ring_start(IoRing* p, u32 rate) {
  return ENOTSUP;
}

static void io_ring_free(IoRing* p) {
  free(p);
}

#endif
#endif

#ifdef CID_AUDIO_OPEN
Term audio_open_run(Env e, Term* f, IoWork* w) {
  u32     rate = (u32)f[0];
  IoRing* p    = io_mem(calloc(1, sizeof *p));
  u32     code = rate < 8000 || rate > 192000 ? EINVAL : io_ring_start(p, rate);
  if (code != 0) {
    io_ring_free(p);
    return io_fail(e, code, code == EINVAL ? NULL : "Audio.open: no audio output");
  }
  return io_done(e, io_hand((intptr_t)p));
}

static void __attribute__((constructor)) audio_open_use(void) {
  io_eff(CID_AUDIO_OPEN, audio_open_run, 0);
}
#endif

#ifdef CID_AUDIO_WRITE
// The samples (interleaved L R ...) into the ring; the frames queued
// after the write. Past the ring's room, the samples are dropped and
// the queue answered as it is.
Term audio_write_run(Env e, Term* f, IoWork* w) {
  IoRing* p   = (IoRing*)(uintptr_t)io_hand_v(f[0]);
  float   pcm[IO_RING * 2];
  u32     n   = 0;
  Term    s   = f[1];
  while (term_aux(s) == CID_CON) {
    Term fb[2];
    spare_free(e, cls_fit(2), ctr_take(e, s, 2, fb));
    if (n < IO_RING * 2) {
      pcm[n] = f32_unbox(fb[0]);
    }
    n += 1;
    s  = fb[1];
  }
  u64 q = n > IO_RING * 2 ? io_ring_write(p, pcm, 0)
    : io_ring_write(p, pcm, n / 2);
  return io_tup(e, f[0], q);
}

static void __attribute__((constructor)) audio_write_use(void) {
  io_eff(CID_AUDIO_WRITE, audio_write_run, 0);
}
#endif

#ifdef CID_AUDIO_CLOSE
Term audio_close_run(Env e, Term* f, IoWork* w) {
  io_ring_free((IoRing*)(uintptr_t)io_hand_v(f[0]));
  return term_pak(CID_UNIT, 0);
}

static void __attribute__((constructor)) audio_close_use(void) {
  io_eff(CID_AUDIO_CLOSE, audio_close_run, 0);
}
#endif
