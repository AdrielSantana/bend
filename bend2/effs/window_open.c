// Window
// ======

#ifdef __OBJC__

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

@interface BendView : NSView <NSWindowDelegate> {
  @public
  NSMutableData* evs;
  u64            flags;
}
@end

@implementation BendView

- (CALayer*)makeBackingLayer {
  return [CAMetalLayer layer];
}

- (BOOL)acceptsFirstResponder {
  return YES;
}

- (BOOL)acceptsFirstMouse:(NSEvent*)ev {
  return YES;
}

- (BOOL)isFlipped {
  return YES;
}

- (void)push:(u32)kind a:(u32)a b:(u32)b c:(u32)c d:(u32)d {
  u32 ev[5] = { kind, a, b, c, d };
  [evs appendBytes:ev length:sizeof ev];
}

- (void)key:(NSEvent*)ev down:(BOOL)down {
  NSString* s = [ev.charactersIgnoringModifiers lowercaseString];
  u32 code = s.length > 0 ? [s characterAtIndex:0] : 65536 + ev.keyCode;
  [self push:0 a:code b:down c:0 d:0];
}

- (void)keyDown:(NSEvent*)ev {
  [self key:ev down:YES];
}

- (void)keyUp:(NSEvent*)ev {
  [self key:ev down:NO];
}

- (void)flagsChanged:(NSEvent*)ev {
  u64 now = ev.modifierFlags;
  [self push:0 a:65536 + ev.keyCode b:(now & ~flags) != 0 c:0 d:0];
  flags = now;
}

- (NSPoint)at:(NSEvent*)ev {
  CGSize  size = ((CAMetalLayer*)self.layer).drawableSize;
  NSPoint p    = [self convertPoint:ev.locationInWindow fromView:nil];
  return NSMakePoint(fmax(0, fmin(floor(p.x), size.width - 1)),
    fmax(0, fmin(floor(p.y), size.height - 1)));
}

- (void)mouse:(NSEvent*)ev down:(BOOL)down {
  NSPoint p = [self at:ev];
  [self push:1 a:p.x b:p.y c:(u32)ev.buttonNumber d:down];
}

- (void)move:(NSEvent*)ev {
  NSPoint p = [self at:ev];
  [self push:2 a:p.x b:p.y c:0 d:0];
}

- (void)mouseDown:(NSEvent*)ev {
  [self mouse:ev down:YES];
}

- (void)mouseUp:(NSEvent*)ev {
  [self mouse:ev down:NO];
}

- (void)rightMouseDown:(NSEvent*)ev {
  [self mouse:ev down:YES];
}

- (void)rightMouseUp:(NSEvent*)ev {
  [self mouse:ev down:NO];
}

- (void)otherMouseDown:(NSEvent*)ev {
  [self mouse:ev down:YES];
}

- (void)otherMouseUp:(NSEvent*)ev {
  [self mouse:ev down:NO];
}

- (void)mouseMoved:(NSEvent*)ev {
  [self move:ev];
}

- (void)mouseDragged:(NSEvent*)ev {
  [self move:ev];
}

- (void)rightMouseDragged:(NSEvent*)ev {
  [self move:ev];
}

- (void)otherMouseDragged:(NSEvent*)ev {
  [self move:ev];
}

- (BOOL)windowShouldClose:(NSWindow*)sender {
  [self push:3 a:0 b:0 c:0 d:0];
  return NO;
}

@end

static id<MTLDevice> window_dev;

static u32 window_make(const char* title, u32 w, u32 h, intptr_t* out,
  const char** why) {
  if (NSScreen.screens.count == 0) {
    *why = "Window.open: no display (build a native binary with bend <file> -o <out> and run it from a desktop session)";
    return ENOTSUP;
  }
  if (window_dev == nil) {
    window_dev = gpu_buf != nil ? gpu_dev : MTLCreateSystemDefaultDevice();
  }
  if (window_dev == nil) {
    *why = "Window.open: no Metal device";
    return ENXIO;
  }
  if (NSApp == nil) {
    [NSApplication sharedApplication];
    NSApp.activationPolicy = NSApplicationActivationPolicyRegular;
    [NSApp finishLaunching];
  }
  @autoreleasepool {
    NSWindow* win = [[NSWindow alloc]
      initWithContentRect:NSMakeRect(0, 0, 1, 1)
      styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        | NSWindowStyleMaskMiniaturizable
      backing:NSBackingStoreBuffered defer:NO];
    win.releasedWhenClosed = NO;
    win.acceptsMouseMovedEvents = YES;
    win.title = [NSString stringWithCString:title
      encoding:NSISOLatin1StringEncoding];
    [win setContentSize:NSMakeSize(w, h)];
    BendView* view = [[BendView alloc] initWithFrame:win.contentLayoutRect];
    view->evs   = [NSMutableData new];
    view->flags = NSEvent.modifierFlags;
    view.wantsLayer = YES;
    CAMetalLayer* layer = (CAMetalLayer*)view.layer;
    layer.device = window_dev;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = NO;
    layer.drawableSize = CGSizeMake(w, h);
    layer.displaySyncEnabled = YES;
    layer.maximumDrawableCount = 2;
    win.contentView = view;
    win.delegate = view;
    [win makeFirstResponder:view];
    [win center];
    [win makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    *out = (intptr_t)CFBridgingRetain(win);
  }
  return 0;
}

#elif defined(__linux__)

// The X11 window: its own connection (so its queue holds only its
// events), the frame's image and the events pumped since the last
// frame, five words each (kind, a, b, c, d) as on the Mac. The same
// block sits in window_frame.c and window_close.c under this guard.
#ifndef BendWin
#define BendWin BendWin
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

typedef struct {
  Display* dpy;
  Window   win;
  Atom     del;
  XImage*  img;
  u32      n;
  u32      cap;
  u32*     evs;
} BendWin;
#endif

static u32 window_make(const char* title, u32 w, u32 h, intptr_t* out,
  const char** why) {
  Display* dpy = XOpenDisplay(NULL);
  if (dpy == NULL) {
    *why = "Window.open: no display (build a native binary with bend <file> -o <out> and run it from a desktop session)";
    return ENOTSUP;
  }
  int scr = DefaultScreen(dpy);
  if (DefaultDepth(dpy, scr) < 24) {
    XCloseDisplay(dpy);
    *why = "Window.open: the display has no 24-bit visual";
    return ENOTSUP;
  }
  BendWin* win = io_mem(calloc(1, sizeof *win));
  win->dpy = dpy;
  win->win = XCreateSimpleWindow(dpy, RootWindow(dpy, scr), 0, 0, w, h, 0, 0,
    BlackPixel(dpy, scr));
  win->del = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
  win->img = XCreateImage(dpy, DefaultVisual(dpy, scr), DefaultDepth(dpy, scr),
    ZPixmap, 0, io_mem(calloc(w * h, 4)), w, h, 32, w * 4);
  win->img->byte_order = LSBFirst;
  XSizeHints hints = { .flags = PMinSize | PMaxSize, .min_width = w,
    .min_height = h, .max_width = w, .max_height = h };
  XSetWMNormalHints(dpy, win->win, &hints);
  XSetWMProtocols(dpy, win->win, &win->del, 1);
  XStoreName(dpy, win->win, title);
  XSelectInput(dpy, win->win, KeyPressMask | KeyReleaseMask | ButtonPressMask
    | ButtonReleaseMask | PointerMotionMask);
  XMapRaised(dpy, win->win);
  XFlush(dpy);
  *out = (intptr_t)win;
  return 0;
}

#elif defined(__EMSCRIPTEN__)
#ifndef BendWin
#define BendWin BendWin
#include <emscripten.h>
typedef struct { u32 w; u32 h; u32* pix; u32 cap; u32* evs; u32 got; } BendWin;
#endif

// The page's <canvas id="bend"> and its listeners, on the main thread: the
// program runs on a worker. Events are the Mac's five words, its key codes
// (a character in lower case, a function key's private-use character,
// 65536 + a modifier's key code) and buttons (0 left, 1 right, 2 middle).
EM_JS(void, window_js_open, (const char* title, u32 w, u32 h), {
  var c = document.getElementById("bend");
  c.width  = w;
  c.height = h;
  Module.bendCtx = c.getContext("2d");
  Module.bendImg = new ImageData(w, h);
  document.title = UTF8ToString(title);
  var evs = Module.bendEvs;
  if (!evs) {
    evs = Module.bendEvs = [];
    var put = function(k, a, b, c, d) {
      if (evs.length < 5120) {
        evs.push(k, a, b, c, d);
      }
    };
    var keys = { Escape: 27, Enter: 13, Tab: 9, Backspace: 127,
      ArrowUp: 63232, ArrowDown: 63233, ArrowLeft: 63234, ArrowRight: 63235,
      Insert: 63271, Delete: 63272, Home: 63273, End: 63275, PageUp: 63276,
      PageDown: 63277, MetaRight: 65590, MetaLeft: 65591, ShiftLeft: 65592,
      CapsLock: 65593, AltLeft: 65594, ControlLeft: 65595, ShiftRight: 65596,
      AltRight: 65597, ControlRight: 65598 };
    var key = function(ev, down) {
      var k = ev.key;
      var f = /^F([0-9]+)$/.exec(k);
      var code = keys[ev.code] || keys[k] || (f ? 63235 + Number(f[1])
        : k.length === 1 ? k.toLowerCase().codePointAt(0) : 65536 + ev.keyCode);
      put(0, code, down, 0, 0);
      if (!ev.metaKey && !ev.ctrlKey && !f) {
        ev.preventDefault();
      }
    };
    var at = function(ev) {
      var r = c.getBoundingClientRect();
      var x = Math.floor((ev.clientX - r.left) * c.width / r.width);
      var y = Math.floor((ev.clientY - r.top) * c.height / r.height);
      return [Math.max(0, Math.min(x, c.width - 1)),
        Math.max(0, Math.min(y, c.height - 1))];
    };
    var mouse = function(ev, down) {
      var p = at(ev);
      if (ev.button < 3) {
        put(1, p[0], p[1], [0, 2, 1][ev.button], down);
      }
    };
    window.addEventListener("keydown", function(ev) { key(ev, 1); });
    window.addEventListener("keyup", function(ev) { key(ev, 0); });
    c.addEventListener("mousedown", function(ev) { mouse(ev, 1); });
    window.addEventListener("mouseup", function(ev) { mouse(ev, 0); });
    c.addEventListener("mousemove", function(ev) {
      var p = at(ev);
      put(2, p[0], p[1], 0, 0);
    });
    c.addEventListener("contextmenu", function(ev) { ev.preventDefault(); });
  }
  evs.length = 0;
});

// A frame is taken off the heap when handed and shown on the display's
// next tick (a hidden tab has no ticks). A program a frame ahead of the
// display waits for that tick, as the Mac's nextDrawable waits for a free
// drawable, and no longer: a frame late for a tick does not wait for the
// next one.
EM_JS(void, window_js_show, (u32* pix, u32 w, u32 h, u32* evs, u32 cap,
  u32* got), {
  var take = function() {
    Module.bendImg.data.set(HEAPU8.subarray(pix, pix + w * h * 4));
    var q = Module.bendEvs;
    var n = Math.min(q.length / 5, cap);
    HEAPU32.set(q.splice(0, n * 5), evs >> 2);
    Atomics.store(HEAP32, got >> 2, n + 1);
    Atomics.notify(HEAP32, got >> 2);
    var show = Module.bendDue = function() {
      var next = Module.bendNext;
      Module.bendCtx.putImageData(Module.bendImg, 0, 0);
      Module.bendFrames = (Module.bendFrames | 0) + 1;
      Module.bendDue = Module.bendNext = null;
      if (next) {
        next();
      }
    };
    if (document.hidden) {
      setTimeout(show, 16);
    } else {
      requestAnimationFrame(show);
    }
  };
  if (Module.bendDue) {
    Module.bendNext = take;
  } else {
    take();
  }
});

static u32 window_make(const char* title, u32 w, u32 h, intptr_t* out,
  const char** why) {
  MAIN_THREAD_EM_ASM({ window_js_open($0, $1, $2); }, title, w, h);
  BendWin* win = io_mem(calloc(1, sizeof *win));
  win->w   = w;
  win->h   = h;
  win->pix = io_mem(calloc((u64)w * h, 4));
  win->cap = 1024;
  win->evs = io_mem(calloc(win->cap * 5, 4));
  *out = (intptr_t)win;
  return 0;
}

#else

static u32 window_make(const char* title, u32 w, u32 h, intptr_t* out,
  const char** why) {
  *why = "Window.open: no display (build a native binary with bend <file> -o <out> and run it from a desktop session)";
  return ENOTSUP;
}

#endif

#ifndef __METAL_VERSION__
// the quadtree level that holds w x h, and a frame filled by the host
INLINE u32 window_k(u32 w, u32 h) {
  u32 m = (w > h ? w : h) - 1;
  return m ? 32 - CLZ(m) : 0;
}

INLINE void window_host(Corpus H, Term root, u32 w, u32 h, u32 k, u32* out) {
  for (u32 i = 0; i < w * h; i += 1) {
    out[i] = window_pix(H, root, k, i % w, i / w);
  }
}
#endif

Term window_open_run(Env e, Term* f, IoWork* w) {
  uint64_t n = 0;
  char* title = io_cstr(e, f[0], &n);
  intptr_t out;
  const char* why = NULL;
  u32 wd = (u32)f[1];
  u32 ht = (u32)f[2];
  u32 q = io_nul(title, n) ? EILSEQ
    : wd < 1 || ht < 1 || wd > 16384 || ht > 16384 ? EINVAL
    : window_make(title, wd, ht, &out, &why);
  free(title);
  if (q != 0) {
    return io_fail(e, q, why);
  }
  return io_done(e, io_hand(out));
}

static void __attribute__((constructor)) window_open_use(void) {
  io_eff(CID_WINDOW_OPEN, window_open_run, 0);
}
