// fl_model.h - the HAR MLP (64 -> 32 -> 6, ReLU, cross-entropy) and plain SGD.
//
// Pure C++ (no Arduino dependencies) so it compiles on the host for tests.
// Mirrors demo/fldemo/model.py exactly: parameters are kept flat in
// TENSOR_SPECS order (fc1.weight 32x64 row-major, fc1.bias 32, fc2.weight 6x32,
// fc2.bias 6) = 2278 float32 = 9112 bytes little-endian on the wire.
//
// Training data is int8 (value * 127) with a float scale, read through
// FL_READ_I8 so it can live in flash (PROGMEM) on the ESP32.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifndef FL_READ_I8
#if defined(ARDUINO)
#include <pgmspace.h>
#define FL_READ_I8(p) ((int8_t)pgm_read_byte(p))
#else
#define FL_READ_I8(p) (*(const int8_t*)(p))
#endif
#endif

namespace fl {

constexpr int NF = 64;  // input features
constexpr int NH = 32;  // hidden units
constexpr int NC = 6;   // classes
constexpr int OFF_W1 = 0;
constexpr int OFF_B1 = OFF_W1 + NH * NF;  // 2048
constexpr int OFF_W2 = OFF_B1 + NH;       // 2080
constexpr int OFF_B2 = OFF_W2 + NC * NH;  // 2272
constexpr int PARAM_COUNT = OFF_B2 + NC;  // 2278
constexpr size_t MODEL_BYTES = PARAM_COUNT * 4;  // 9112

struct Params {
  float p[PARAM_COUNT];
  float* w1() { return p + OFF_W1; }
  float* b1() { return p + OFF_B1; }
  float* w2() { return p + OFF_W2; }
  float* b2() { return p + OFF_B2; }
  const float* w1() const { return p + OFF_W1; }
  const float* b1() const { return p + OFF_B1; }
  const float* w2() const { return p + OFF_W2; }
  const float* b2() const { return p + OFF_B2; }
};

// ---------------------------------------------------------------- (de)serialisation
inline bool load_bytes(Params& m, const uint8_t* buf, size_t len) {
  if (len != MODEL_BYTES) return false;
  for (int i = 0; i < PARAM_COUNT; i++) {
    uint32_t u = (uint32_t)buf[4 * i] | ((uint32_t)buf[4 * i + 1] << 8) |
                 ((uint32_t)buf[4 * i + 2] << 16) | ((uint32_t)buf[4 * i + 3] << 24);
    memcpy(&m.p[i], &u, 4);
  }
  return true;
}

inline void store_float_le(uint8_t* out, float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  out[0] = (uint8_t)u; out[1] = (uint8_t)(u >> 8); out[2] = (uint8_t)(u >> 16); out[3] = (uint8_t)(u >> 24);
}

inline void to_bytes(const Params& m, uint8_t* out) {
  for (int i = 0; i < PARAM_COUNT; i++) store_float_le(out + 4 * i, m.p[i]);
}

inline void scale_params(Params& m, float k) {
  for (int i = 0; i < PARAM_COUNT; i++) m.p[i] *= k;
}

// ---------------------------------------------------------------- forward
// x: NF floats. pre (optional): NH pre-activations. h: NH post-ReLU. logits: NC.
inline void forward_f(const Params& m, const float* x, float* pre, float* h, float* logits) {
  const float* W1 = m.w1();
  const float* bias1 = m.b1();
  for (int j = 0; j < NH; j++) {
    float a = bias1[j];
    const float* row = W1 + j * NF;
    for (int i = 0; i < NF; i++) a += row[i] * x[i];
    if (pre) pre[j] = a;
    h[j] = a > 0.f ? a : 0.f;
  }
  const float* W2 = m.w2();
  const float* bias2 = m.b2();
  for (int c = 0; c < NC; c++) {
    float a = bias2[c];
    const float* row = W2 + c * NH;
    for (int j = 0; j < NH; j++) a += row[j] * h[j];
    logits[c] = a;
  }
}

// Dequantise one int8 sample row (possibly in flash).
inline void load_row(const int8_t* X, int s, float scale, float* x) {
  const int8_t* r = X + (size_t)s * NF;
  for (int i = 0; i < NF; i++) x[i] = (float)FL_READ_I8(r + i) * scale;
}

// Cross-entropy for one sample; fills probs (softmax). Returns -log p[y].
inline float softmax_xent(const float* logits, int y, float* probs) {
  float mx = logits[0];
  for (int c = 1; c < NC; c++) if (logits[c] > mx) mx = logits[c];
  float sum = 0.f;
  for (int c = 0; c < NC; c++) { probs[c] = expf(logits[c] - mx); sum += probs[c]; }
  float inv = 1.f / sum;
  for (int c = 0; c < NC; c++) probs[c] *= inv;
  return (logf(sum) + mx) - logits[y];  // logsumexp - logit_y
}

// ---------------------------------------------------------------- SGD
// One epoch of mini-batch SGD (no momentum), mean cross-entropy per batch,
// identical to torch.optim.SGD + nn.CrossEntropyLoss in fldemo.model.train_sgd.
//   order: sample visiting order (length n) or nullptr for 0..n-1 (deterministic).
//   grad:  scratch Params (caller-provided, avoids a 9 KB stack frame).
//   yield: optional callback after each batch (e.g. feed the watchdog / OLED).
// Returns the sample-weighted mean loss of the epoch (same as train_sgd).
typedef void (*BatchHook)(int done, int total, void* ctx);

inline float sgd_epoch(Params& m, Params& grad, const int8_t* X, const uint8_t* Y, int n,
                       float scale, float lr, int batch, const uint16_t* order,
                       BatchHook hook = nullptr, void* ctx = nullptr) {
  float x[NF], pre[NH], h[NH], logits[NC], probs[NC], dz[NC], dh[NH];
  double total = 0.0;
  if (batch < 1) batch = 1;
  for (int start = 0; start < n; start += batch) {
    int end = start + batch < n ? start + batch : n;
    int bs = end - start;
    float invbs = 1.f / (float)bs;
    memset(grad.p, 0, sizeof(grad.p));
    float batch_loss = 0.f;
    for (int k = start; k < end; k++) {
      int s = order ? (int)order[k] : k;
      int y = (int)Y[s];
      load_row(X, s, scale, x);
      forward_f(m, x, pre, h, logits);
      batch_loss += softmax_xent(logits, y, probs);
      for (int c = 0; c < NC; c++) dz[c] = (probs[c] - (c == y ? 1.f : 0.f)) * invbs;
      for (int j = 0; j < NH; j++) dh[j] = 0.f;
      float* gW2 = grad.w2();
      float* gB2 = grad.b2();
      const float* W2 = m.w2();
      for (int c = 0; c < NC; c++) {
        gB2[c] += dz[c];
        for (int j = 0; j < NH; j++) {
          gW2[c * NH + j] += dz[c] * h[j];
          dh[j] += W2[c * NH + j] * dz[c];
        }
      }
      float* gW1 = grad.w1();
      float* gB1 = grad.b1();
      for (int j = 0; j < NH; j++) {
        if (pre[j] <= 0.f) continue;  // ReLU'(0) = 0, as in PyTorch
        float d = dh[j];
        gB1[j] += d;
        float* row = gW1 + j * NF;
        for (int i = 0; i < NF; i++) row[i] += d * x[i];
      }
    }
    for (int i = 0; i < PARAM_COUNT; i++) m.p[i] -= lr * grad.p[i];
    total += (double)(batch_loss * invbs) * bs;  // torch: float(mean_loss) * len(b)
    if (hook) hook(end, n, ctx);
  }
  return n > 0 ? (float)(total / n) : 0.f;
}

// Fisher-Yates shuffle with a caller-supplied 32-bit RNG.
typedef uint32_t (*Rand32)();
inline void shuffle_order(uint16_t* order, int n, Rand32 rnd) {
  for (int i = 0; i < n; i++) order[i] = (uint16_t)i;
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(rnd() % (uint32_t)(i + 1));
    uint16_t t = order[i]; order[i] = order[j]; order[j] = t;
  }
}

}  // namespace fl
