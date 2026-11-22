// fl_b64.h - standard base64 (RFC 4648, '=' padding). Host-testable.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace fl {

inline size_t b64_len(size_t n) { return 4 * ((n + 2) / 3); }

// Writes b64_len(n) chars plus a NUL. Returns the number of chars (without NUL).
inline size_t b64_encode(const uint8_t* in, size_t n, char* out) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  size_t o = 0, i = 0;
  for (; i + 3 <= n; i += 3) {
    uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8) | in[i + 2];
    out[o++] = T[v >> 18]; out[o++] = T[(v >> 12) & 63]; out[o++] = T[(v >> 6) & 63]; out[o++] = T[v & 63];
  }
  if (n - i == 1) {
    uint32_t v = (uint32_t)in[i] << 16;
    out[o++] = T[v >> 18]; out[o++] = T[(v >> 12) & 63]; out[o++] = '='; out[o++] = '=';
  } else if (n - i == 2) {
    uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8);
    out[o++] = T[v >> 18]; out[o++] = T[(v >> 12) & 63]; out[o++] = T[(v >> 6) & 63]; out[o++] = '=';
  }
  out[o] = 0;
  return o;
}

inline int b64_val(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

// Strict decode (length multiple of 4, padding only at the end).
// Returns decoded length, or -1 on malformed input / output overflow.
inline long b64_decode(const char* in, size_t n, uint8_t* out, size_t cap) {
  if (n % 4) return -1;
  size_t o = 0;
  for (size_t i = 0; i < n; i += 4) {
    int v[4], pad = 0;
    for (int k = 0; k < 4; k++) {
      char c = in[i + k];
      if (c == '=') {
        if (i + 4 != n || k < 2) return -1;
        v[k] = 0; pad++;
      } else {
        if (pad) return -1;
        v[k] = b64_val(c);
        if (v[k] < 0) return -1;
      }
    }
    uint32_t w = ((uint32_t)v[0] << 18) | ((uint32_t)v[1] << 12) | ((uint32_t)v[2] << 6) | (uint32_t)v[3];
    int bytes = 3 - pad;
    if (o + bytes > cap) return -1;
    out[o++] = (uint8_t)(w >> 16);
    if (bytes > 1) out[o++] = (uint8_t)(w >> 8);
    if (bytes > 2) out[o++] = (uint8_t)w;
  }
  return (long)o;
}

}  // namespace fl
