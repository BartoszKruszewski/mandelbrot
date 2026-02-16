/* =============================================================
   Mandelbrot Explorer – High-Performance WebGL2 Engine
   =============================================================
   • WebGL2 fragment-shader Mandelbrot (emulated double precision)
   • Adaptive resolution: low-res while interacting → hi-res on idle
   • Arbitrary-precision coordinate tracking via Decimal.js
   • Cardioid / bulb-2 early bail-out
   • GPU tiled progressive hi-res render (scissor-based, visible)
   • Smooth cosine-based cyclic colour palette
   • Iteration count scales with zoom depth
   ============================================================= */

"use strict";

/* ----------------------------------------------------------
   0.  CONSTANTS & CONFIG
   ---------------------------------------------------------- */
const LOW_RES_DIVISOR  = 4;
const IDLE_DEBOUNCE_MS = 300;
const HI_RES_TILE_SIZE = 128;       // px per tile edge
const BASE_ITER_LOW    = 200;
const BASE_ITER_HIGH   = 800;
const ZOOM_SPEED       = 1.1;
const TILES_PER_FRAME  = 4;         // GPU tiles per rAF yield

/* ----------------------------------------------------------
   1.  DOM REFS
   ---------------------------------------------------------- */
const canvas      = document.getElementById("fractalCanvas");
const zoomDisp    = document.getElementById("zoomDisplay");
const realDisp    = document.getElementById("realDisplay");
const imagDisp    = document.getElementById("imagDisplay");
const progressBar = document.getElementById("progressBar");
const progressTxt = document.getElementById("progressText");

/* ----------------------------------------------------------
   2.  STATE
   ---------------------------------------------------------- */
let idleTimer    = null;
let hiResAbortId = 0;

const state = {
  centerRe:    new Decimal("-0.5"),
  centerIm:    new Decimal("0.0"),
  zoom:        new Decimal("1"),
  pixelScale:  3,
  dragging:    false,
  dragX:       0,
  dragY:       0,
  interacting: false,
  hiResDone:   false,
};

function floatCenter() {
  return { re: state.centerRe.toNumber(), im: state.centerIm.toNumber() };
}

/** Scale as Decimal (never loses precision) */
function decimalScale() {
  return new Decimal(state.pixelScale).div(state.zoom);
}

/** Scale as JS float — only for shader uniforms (may underflow to 0 at extreme zoom) */
function floatScale() {
  return decimalScale().toNumber();
}

/** Adaptive iteration count — grows with log of zoom (Decimal-safe) */
function iterCount(base) {
  /* Decimal.log2 works for arbitrarily large zoom values */
  const log2z = state.zoom.ln().div(Math.LN2).toNumber();
  const extra = Math.max(0, log2z) * 60;
  return Math.min(Math.round(base + extra), 4000);
}

/* ----------------------------------------------------------
   3.  CANVAS SIZE
   ---------------------------------------------------------- */
let W, H, dpr;

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + "px";
  canvas.style.height = H + "px";
  onInteractionStart();
  renderLowRes();
  updateUI();
  onInteractionEnd();
}
window.addEventListener("resize", resize);

/* ----------------------------------------------------------
   4.  WEBGL2 SETUP
   ---------------------------------------------------------- */
const gl = canvas.getContext("webgl2", {
  antialias: false,
  depth: false,
  stencil: false,
  premultipliedAlpha: false,
  preserveDrawingBuffer: true,       // keep pixels between frames
});
if (!gl) { alert("WebGL 2 required."); throw new Error("No WebGL2"); }

/* ---- Shaders ---- */
const VERT_SRC = `#version 300 es
precision highp float;
out vec2 vUV;
void main(){
  float x = float((gl_VertexID & 1) << 2);
  float y = float((gl_VertexID & 2) << 1);
  vUV = vec2(x * 0.5, y * 0.5);
  gl_Position = vec4(x - 1.0, y - 1.0, 0.0, 1.0);
}`;

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;

uniform vec2  u_resolution;
uniform float u_centerHi_re;
uniform float u_centerLo_re;
uniform float u_centerHi_im;
uniform float u_centerLo_im;
uniform float u_scale;
uniform int   u_maxIter;
uniform int   u_paletteMode;

vec2 ds_add(vec2 a, vec2 b) {
  float t1 = a.x + b.x;
  float e  = t1 - a.x;
  float t2 = ((b.x - e) + (a.x - (t1 - e))) + a.y + b.y;
  float s  = t1 + t2;
  return vec2(s, t2 - (s - t1));
}
vec2 ds_mul(vec2 a, vec2 b) {
  float t1 = a.x * b.x;
  float t2 = (a.x * b.x - t1) + a.x * b.y + a.y * b.x;
  float s  = t1 + t2;
  return vec2(s, t2 - (s - t1));
}

vec3 palette(float t) {
  if (u_paletteMode == 0) {
    /* Cosmos – deep blue/cyan only (no yellow) */
    vec3 a = vec3(0.10, 0.20, 0.40);
    vec3 b = vec3(0.18, 0.35, 0.60);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.00, 0.18, 0.35);
    vec3 col = a + b * cos(6.28318 * (c * t + d));
    col.r *= 0.55;              /* suppress warm tones */
    col.g *= 0.35;              /* remove green, push to purple */
    float glow = pow(max(col.b, 0.0), 2.2) * 0.3;
    return clamp(col + vec3(glow * 0.4, 0.0, glow), 0.0, 1.0);
  } else if (u_paletteMode == 1) {
    /* Noir – B&W with strong 3D emboss/relief effect */
    float base = 0.5 + 0.5 * cos(6.28318 * t * 0.8);
    /* multi-frequency edges for depth */
    float edge1 = 0.5 + 0.5 * cos(6.28318 * t * 3.0 + 0.8);
    float edge2 = 0.5 + 0.5 * cos(6.28318 * t * 7.0 + 2.0);
    float edge3 = 0.5 + 0.5 * sin(6.28318 * t * 5.0);
    /* specular highlight */
    float spec = pow(edge1, 5.0) * 0.6;
    /* ambient occlusion shadow */
    float ao = (1.0 - edge2) * 0.25;
    /* rim light */
    float rim = pow(edge3, 4.0) * 0.3;
    float lum = clamp(base * 0.7 + spec + rim - ao + 0.1, 0.0, 1.0);
    /* slight warm/cool tint for depth perception */
    float warm = spec * 0.08;
    return clamp(vec3(lum + warm, lum, lum - warm * 0.5), 0.0, 1.0);
  } else {
    /* Inferno – inverted Noir (white base) with red accents */
    float base = 0.5 + 0.5 * cos(6.28318 * t * 0.8);
    float edge1 = 0.5 + 0.5 * cos(6.28318 * t * 3.0 + 0.8);
    float edge2 = 0.5 + 0.5 * cos(6.28318 * t * 7.0 + 2.0);
    float edge3 = 0.5 + 0.5 * sin(6.28318 * t * 5.0);
    float spec = pow(edge1, 5.0) * 0.55;
    float ao = (1.0 - edge2) * 0.22;
    float rim = pow(edge3, 4.0) * 0.25;
    float lum = clamp(base * 0.7 + spec + rim - ao + 0.1, 0.0, 1.0);
    float inv = clamp(1.0 - lum * 0.85, 0.0, 1.0);
    float red = pow(edge1, 3.0) * 0.35 + pow(edge3, 4.0) * 0.15;
    return clamp(vec3(inv + red, inv * 0.95, inv * 0.95), 0.0, 1.0);
  }
}

void main() {
  float aspect = u_resolution.x / u_resolution.y;
  float px = (vUV.x - 0.5) * aspect;
  float py = -(vUV.y - 0.5);

  vec2 c_re = ds_add(vec2(u_centerHi_re, u_centerLo_re), vec2(px * u_scale, 0.0));
  vec2 c_im = ds_add(vec2(u_centerHi_im, u_centerLo_im), vec2(py * u_scale, 0.0));

  float crf = c_re.x, cif = c_im.x;
  float q = (crf - 0.25) * (crf - 0.25) + cif * cif;
  if ((q * (q + (crf - 0.25))) <= (0.25 * cif * cif)) {
    fragColor = (u_paletteMode == 2) ? vec4(1,1,1,1) : vec4(0,0,0,1); return;
  }
  if (((crf + 1.0) * (crf + 1.0) + cif * cif) <= 0.0625) {
    fragColor = (u_paletteMode == 2) ? vec4(1,1,1,1) : vec4(0,0,0,1); return;
  }

  vec2 z_re = vec2(0.0), z_im = vec2(0.0);
  int iter = 0;
  float zrsq, zisq;
  for (int i = 0; i < 4000; i++) {
    if (i >= u_maxIter) break;
    vec2 zr2 = ds_mul(z_re, z_re);
    vec2 zi2 = ds_mul(z_im, z_im);
    zrsq = zr2.x; zisq = zi2.x;
    if (zrsq + zisq > 4.0) break;
    vec2 prod = ds_mul(z_re, z_im);
    z_im = ds_add(ds_add(prod, prod), c_im);
    z_re = ds_add(ds_add(zr2, vec2(-zi2.x, -zi2.y)), c_re);
    iter++;
  }

  if (iter >= u_maxIter) {
    fragColor = (u_paletteMode == 2) ? vec4(1,1,1,1) : vec4(0,0,0,1);
  } else {
    float mag2 = zrsq + zisq;
    float nu   = log(log(max(mag2, 1.0))) / log(2.0);
    float t    = (float(iter) + 1.0 - nu) / 80.0;
    fragColor  = vec4(palette(t), 1.0);
  }
}`;

function compileShader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
}
function linkProgram(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(p));
    return null;
  }
  return p;
}

const vShader = compileShader(gl.VERTEX_SHADER,  VERT_SRC);
const fShader = compileShader(gl.FRAGMENT_SHADER, FRAG_SRC);
const prog    = linkProgram(vShader, fShader);
gl.useProgram(prog);

const u_resolution  = gl.getUniformLocation(prog, "u_resolution");
const u_centerHi_re = gl.getUniformLocation(prog, "u_centerHi_re");
const u_centerLo_re = gl.getUniformLocation(prog, "u_centerLo_re");
const u_centerHi_im = gl.getUniformLocation(prog, "u_centerHi_im");
const u_centerLo_im = gl.getUniformLocation(prog, "u_centerLo_im");
const u_scale       = gl.getUniformLocation(prog, "u_scale");
const u_maxIter     = gl.getUniformLocation(prog, "u_maxIter");
const u_paletteMode = gl.getUniformLocation(prog, "u_paletteMode");

/* ----------------------------------------------------------
   PALETTE PROFILES & REVOLVER SWITCHER
   ---------------------------------------------------------- */
const PALETTES = [
  { id: 0, name: "Cosmos",  theme: "theme-cosmos"  },
  { id: 1, name: "Noir",    theme: "theme-noir"    },
  { id: 2, name: "Inferno", theme: "theme-inferno" },
];
let currentPalette = 0;

function applyPalette(index, animate = true) {
  currentPalette = index;
  const pal = PALETTES[index];
  const toolbar = document.getElementById("toolbar");
  const nameEl  = document.getElementById("paletteName");
  const cylinder = document.querySelector(".revolver-cylinder");

  /* Update toolbar theme class */
  toolbar.className = pal.theme;

  /* Update label */
  nameEl.textContent = pal.name;

  /* Spin the cylinder */
  if (animate && cylinder) {
    cylinder.classList.remove("spin");
    /* reset animation */
    void cylinder.getBoundingClientRect();
    cylinder.classList.add("spin");
    cylinder.addEventListener("animationend", () => {
      cylinder.classList.remove("spin");
    }, { once: true });
  }

  /* Re-render with new palette */
  onInteractionStart();
  requestRender();
  onInteractionEnd();
}

/* Revolver button click handler */
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("paletteBtn");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const next = (currentPalette + 1) % PALETTES.length;
      applyPalette(next, true);
    });
  }
  /* Apply initial theme */
  document.getElementById("toolbar").classList.add("theme-cosmos");
});

/* Blit shader for FBO upscale */
const BLIT_VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main(){
  float x = float((gl_VertexID & 1) << 2);
  float y = float((gl_VertexID & 2) << 1);
  vUV = vec2(x * 0.5, y * 0.5);
  gl_Position = vec4(x - 1.0, y - 1.0, 0.0, 1.0);
}`;
const BLIT_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D u_tex;
out vec4 fragColor;
void main(){ fragColor = texture(u_tex, vUV); }`;
const blitVS   = compileShader(gl.VERTEX_SHADER,   BLIT_VERT);
const blitFS   = compileShader(gl.FRAGMENT_SHADER,  BLIT_FRAG);
const blitProg = linkProgram(blitVS, blitFS);
const u_blitTex = gl.getUniformLocation(blitProg, "u_tex");

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

/* ----------------------------------------------------------
   5.  DOUBLE-FLOAT SPLIT
   ---------------------------------------------------------- */
function splitDouble(v) {
  const hi = Math.fround(v);
  return [hi, v - hi];
}

/* ----------------------------------------------------------
   6.  LOW-RES FBO + RENDER
   ---------------------------------------------------------- */
let lowResFBO = null, lowResTex = null, lowResW = 0, lowResH = 0;

function ensureLowResFBO(w, h) {
  if (lowResFBO && lowResW === w && lowResH === h) return;
  if (lowResFBO) { gl.deleteFramebuffer(lowResFBO); gl.deleteTexture(lowResTex); }
  lowResW = w; lowResH = h;
  lowResTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, lowResTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  lowResFBO = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, lowResFBO);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, lowResTex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function setMandelbrotUniforms(w, h, maxIter) {
  const c = floatCenter();
  const s = floatScale();
  const [reHi, reLo] = splitDouble(c.re);
  const [imHi, imLo] = splitDouble(c.im);
  gl.useProgram(prog);
  gl.bindVertexArray(vao);
  gl.uniform2f(u_resolution, w, h);
  gl.uniform1f(u_centerHi_re, reHi);
  gl.uniform1f(u_centerLo_re, reLo);
  gl.uniform1f(u_centerHi_im, imHi);
  gl.uniform1f(u_centerLo_im, imLo);
  gl.uniform1f(u_scale, s);
  gl.uniform1i(u_maxIter, maxIter);
  gl.uniform1i(u_paletteMode, currentPalette);
}

/** Render low-res into FBO, then blit to full-screen canvas */
function renderLowRes() {
  const fullW = canvas.width;
  const fullH = canvas.height;
  const w = Math.max(1, Math.round(fullW / LOW_RES_DIVISOR));
  const h = Math.max(1, Math.round(fullH / LOW_RES_DIVISOR));

  ensureLowResFBO(w, h);

  /* Render Mandelbrot to FBO at low resolution */
  gl.bindFramebuffer(gl.FRAMEBUFFER, lowResFBO);
  gl.viewport(0, 0, w, h);
  gl.disable(gl.SCISSOR_TEST);
  setMandelbrotUniforms(w, h, iterCount(BASE_ITER_LOW));
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  /* Blit FBO → screen (bilinear upscale fills entire canvas) */
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, fullW, fullH);
  gl.useProgram(blitProg);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, lowResTex);
  gl.uniform1i(u_blitTex, 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

/* ----------------------------------------------------------
   7.  PROGRESSIVE HI-RES (GPU scissor-tiled)
   Each tile is rendered directly to the default framebuffer
   at full resolution via gl.scissor. Since preserveDrawingBuffer
   is true, already-drawn tiles persist.  Between batches we
   yield to the browser via rAF so tiles appear progressively.
   ---------------------------------------------------------- */
function startHiResRender() {
  const myId = ++hiResAbortId;

  const fullW = canvas.width;
  const fullH = canvas.height;
  const maxIter = iterCount(BASE_ITER_HIGH);

  /* Build tile list */
  const tiles = [];
  for (let y = 0; y < fullH; y += HI_RES_TILE_SIZE) {
    for (let x = 0; x < fullW; x += HI_RES_TILE_SIZE) {
      tiles.push({
        x, y,
        w: Math.min(HI_RES_TILE_SIZE, fullW - x),
        h: Math.min(HI_RES_TILE_SIZE, fullH - y),
      });
    }
  }

  const total = tiles.length;
  let done = 0;
  setProgress(0);

  /* Set Mandelbrot uniforms once (constant across all tiles) */
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, fullW, fullH);
  setMandelbrotUniforms(fullW, fullH, maxIter);
  gl.enable(gl.SCISSOR_TEST);

  function renderBatch() {
    if (myId !== hiResAbortId) {
      gl.disable(gl.SCISSOR_TEST);
      return;                          // aborted
    }

    /* Re-bind program/uniforms in case low-res blit changed them */
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, fullW, fullH);
    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.enable(gl.SCISSOR_TEST);

    const batchEnd = Math.min(done + TILES_PER_FRAME, total);
    for (let i = done; i < batchEnd; i++) {
      const t = tiles[i];
      gl.scissor(t.x, t.y, t.w, t.h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    done = batchEnd;
    setProgress(Math.round((done / total) * 100));

    if (done < total) {
      requestAnimationFrame(renderBatch);
    } else {
      gl.disable(gl.SCISSOR_TEST);
      state.hiResDone = true;
    }
  }

  requestAnimationFrame(renderBatch);
}

/* ----------------------------------------------------------
   8.  UI UPDATES
   ---------------------------------------------------------- */
function setProgress(pct) {
  progressBar.style.width = pct + "%";
  progressTxt.textContent = pct + "%";
}

function updateUI() {
  const z = state.zoom;
  zoomDisp.textContent = z.toExponential(2);
  const prec = Math.max(6, Math.min(40,
    Math.ceil(Math.log10(z.toNumber() + 1)) + 4));
  realDisp.textContent = state.centerRe.toSignificantDigits(prec).toString();
  imagDisp.textContent = state.centerIm.toSignificantDigits(prec).toString();
}

/* ----------------------------------------------------------
   9.  INTERACTION LIFECYCLE
   ---------------------------------------------------------- */
function onInteractionStart() {
  state.interacting = true;
  state.hiResDone   = false;
  hiResAbortId++;                     // abort any running hi-res
  clearTimeout(idleTimer);
  setProgress(0);
}

function onInteractionEnd() {
  scheduleIdle();
}

function scheduleIdle() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    state.interacting = false;
    startHiResRender();               // begin progressive tiles
  }, IDLE_DEBOUNCE_MS);
}

function requestRender() {
  renderLowRes();
  updateUI();
}

/* ----------------------------------------------------------
   10. INTERACTION: PAN (pointer)
   ---------------------------------------------------------- */
canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  state.dragging = true;
  state.dragX = e.clientX;
  state.dragY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
  onInteractionStart();
});

canvas.addEventListener("pointermove", (e) => {
  if (!state.dragging) return;
  const dx = e.clientX - state.dragX;
  const dy = e.clientY - state.dragY;
  state.dragX = e.clientX;
  state.dragY = e.clientY;
  const aspect = W / H;
  /* Use Decimal scale to avoid precision loss at extreme zoom */
  const ds = new Decimal(state.pixelScale).div(state.zoom);
  state.centerRe = state.centerRe.minus(ds.times(dx / W * aspect));
  state.centerIm = state.centerIm.minus(ds.times(dy / H));
  requestRender();
});

canvas.addEventListener("pointerup", (e) => {
  state.dragging = false;
  canvas.releasePointerCapture(e.pointerId);
  onInteractionEnd();
});

/* ----------------------------------------------------------
   11. INTERACTION: ZOOM (wheel)
   ---------------------------------------------------------- */
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  onInteractionStart();

  const factor = e.deltaY < 0 ? ZOOM_SPEED : 1 / ZOOM_SPEED;
  const aspect = W / H;
  const ds = decimalScale();

  /* Pixel fractions matching shader mapping:
     shader px = (vUV.x - 0.5) * aspect   → Re offset
     shader py = -(vUV.y - 0.5)           → Im offset
     But drag convention: centerIm -= dy/H * scale
     So effective screen→world for Im: centerIm - (clientY/H - 0.5) * scale
     which equals centerIm + (0.5 - clientY/H) * scale                      */
  const fxScreen = (e.clientX / W - 0.5) * aspect;   // Re fraction
  const fyScreen = (e.clientY / H - 0.5);            // screen Y fraction

  /* World point under cursor */
  const wRe = state.centerRe.plus(ds.times(fxScreen));
  const wIm = state.centerIm.plus(ds.times(fyScreen));

  state.zoom = state.zoom.times(new Decimal(factor));

  /* After zoom, re-centre so the same world point stays under cursor */
  const dsNew = decimalScale();
  state.centerRe = wRe.minus(dsNew.times(fxScreen));
  state.centerIm = wIm.minus(dsNew.times(fyScreen));

  requestRender();
  onInteractionEnd();
}, { passive: false });

/* ----------------------------------------------------------
   12. TOUCH GESTURES
   ---------------------------------------------------------- */
let touchCache = [];
let lastPinchDist = null;

canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  for (const t of e.changedTouches)
    touchCache.push({ id: t.identifier, x: t.clientX, y: t.clientY });
  if (touchCache.length === 1) {
    state.dragging = true;
    state.dragX = touchCache[0].x;
    state.dragY = touchCache[0].y;
  }
  onInteractionStart();
}, { passive: false });

canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  for (const t of e.changedTouches) {
    const idx = touchCache.findIndex((c) => c.id === t.identifier);
    if (idx >= 0) { touchCache[idx].x = t.clientX; touchCache[idx].y = t.clientY; }
  }
  if (touchCache.length === 1 && state.dragging) {
    const dx = touchCache[0].x - state.dragX;
    const dy = touchCache[0].y - state.dragY;
    state.dragX = touchCache[0].x;
    state.dragY = touchCache[0].y;
    const aspect = W / H;
    const ds = new Decimal(state.pixelScale).div(state.zoom);
    state.centerRe = state.centerRe.minus(ds.times(dx / W * aspect));
    state.centerIm = state.centerIm.minus(ds.times(dy / H));
    requestRender();
  } else if (touchCache.length >= 2) {
    const ddx = touchCache[1].x - touchCache[0].x;
    const ddy = touchCache[1].y - touchCache[0].y;
    const dist = Math.hypot(ddx, ddy);
    if (lastPinchDist !== null) {
      const factor = dist / lastPinchDist;
      const cx = (touchCache[0].x + touchCache[1].x) / 2;
      const cy = (touchCache[0].y + touchCache[1].y) / 2;
      const aspect = W / H;
      const ds = decimalScale();
      const fxS = (cx / W - 0.5) * aspect;
      const fyS = (cy / H - 0.5);
      const wRe = state.centerRe.plus(ds.times(fxS));
      const wIm = state.centerIm.plus(ds.times(fyS));
      state.zoom = state.zoom.times(new Decimal(factor));
      const dsNew = decimalScale();
      state.centerRe = wRe.minus(dsNew.times(fxS));
      state.centerIm = wIm.minus(dsNew.times(fyS));
      requestRender();
    }
    lastPinchDist = dist;
  }
}, { passive: false });

canvas.addEventListener("touchend", (e) => {
  for (const t of e.changedTouches)
    touchCache = touchCache.filter((c) => c.id !== t.identifier);
  if (touchCache.length < 2) lastPinchDist = null;
  if (touchCache.length === 0) state.dragging = false;
  onInteractionEnd();
}, { passive: false });

/* ----------------------------------------------------------
   13. KEYBOARD
   ---------------------------------------------------------- */
document.addEventListener("keydown", (e) => {
  const PAN = 0.05;
  const ds = decimalScale();
  let handled = true;
  switch (e.key) {
    case "ArrowLeft":  state.centerRe = state.centerRe.minus(ds.times(PAN)); break;
    case "ArrowRight": state.centerRe = state.centerRe.plus(ds.times(PAN));  break;
    case "ArrowUp":    state.centerIm = state.centerIm.plus(ds.times(PAN));  break;
    case "ArrowDown":  state.centerIm = state.centerIm.minus(ds.times(PAN)); break;
    case "+": case "=": state.zoom = state.zoom.times(new Decimal(ZOOM_SPEED)); break;
    case "-":           state.zoom = state.zoom.div(new Decimal(ZOOM_SPEED));  break;
    case "r": case "R":
      state.centerRe = new Decimal("-0.5");
      state.centerIm = new Decimal("0.0");
      state.zoom     = new Decimal("1");
      break;
    default: handled = false;
  }
  if (handled) { onInteractionStart(); requestRender(); onInteractionEnd(); }
});

/* ----------------------------------------------------------
   14. BOOT
   ---------------------------------------------------------- */
resize();
