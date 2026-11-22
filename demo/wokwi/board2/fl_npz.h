// fl_npz.h - NPZ writer (ZIP with STORED entries of .npy files), byte-identical
// to demo/fldemo/wire.py build_npz_stored() for the HAR model. Host-testable.
//
// Layout: tensor_000000.npy .. tensor_000003.npy (float32 '<f4'), then
// __manifest__.npy (uint8 '|u1' bytes of the compact, key-sorted manifest JSON).
// Fixed DOS time 0 / date 0x0021 (1980-01-01), version 20, no extra fields.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "fl_model.h"

namespace fl {

inline uint32_t crc32_update(uint32_t crc, const uint8_t* p, size_t n) {
  crc = ~crc;
  while (n--) {
    crc ^= *p++;
    for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

struct TensorSpec { const char* name; int rows; int cols; int offset; };  // cols == 0 -> 1-D
static const TensorSpec TENSORS[4] = {
    {"fc1.weight", NH, NF, OFF_W1},
    {"fc1.bias", NH, 0, OFF_B1},
    {"fc2.weight", NC, NH, OFF_W2},
    {"fc2.bias", NC, 0, OFF_B2},
};

// Manifest JSON exactly as json.dumps(..., sort_keys=True, separators=(",", ":")).
inline size_t manifest_json(char* out, size_t cap) {
  size_t o = (size_t)snprintf(out, cap, "{\"format_version\":1,\"tensor_count\":4,\"tensors\":[");
  for (int i = 0; i < 4; i++) {
    const TensorSpec& t = TENSORS[i];
    char shape[24];
    if (t.cols) snprintf(shape, sizeof shape, "[%d,%d]", t.rows, t.cols);
    else snprintf(shape, sizeof shape, "[%d]", t.rows);
    o += (size_t)snprintf(out + o, cap > o ? cap - o : 0,
                          "%s{\"archive_key\":\"tensor_%06d\",\"dtype\":\"<f4\",\"name\":\"%s\",\"shape\":%s}",
                          i ? "," : "", i, t.name, shape);
  }
  o += (size_t)snprintf(out + o, cap > o ? cap - o : 0, "]}");
  return o;
}

// .npy v1.0 header (magic+version+len+dict, padded with spaces + '\n' to a multiple of 64).
// shape_txt like "(32, 64)" or "(32,)". Returns header length.
inline size_t npy_header(uint8_t* out, const char* descr, const char* shape_txt) {
  char dict[128];
  int dl = snprintf(dict, sizeof dict, "{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, shape_txt);
  int pad = 64 - ((10 + dl + 1) % 64);
  if (pad == 64) pad = 0;
  int hl = dl + pad + 1;
  memcpy(out, "\x93NUMPY\x01\x00", 8);
  out[8] = (uint8_t)(hl & 0xFF);
  out[9] = (uint8_t)(hl >> 8);
  memcpy(out + 10, dict, dl);
  memset(out + 10 + dl, ' ', pad);
  out[10 + dl + pad] = '\n';
  return 10 + (size_t)hl;
}

namespace detail {
inline void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
inline void put32(uint8_t* p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
}  // namespace detail

constexpr uint16_t DOS_TIME = 0;
constexpr uint16_t DOS_DATE = (0 << 9) | (1 << 5) | 1;  // 0x0021

// Builds the NPZ into out (capacity cap). Returns its size, or 0 if cap is too small.
// The HAR model always yields NPZ_SIZE bytes.
constexpr size_t NPZ_SIZE = 10687;

inline size_t build_npz(const Params& m, uint8_t* out, size_t cap) {
  using namespace detail;
  struct Ent { char name[24]; uint32_t off, size, crc; };
  Ent ents[5];
  size_t o = 0;
  char manifest[512];
  size_t mlen = manifest_json(manifest, sizeof manifest);

  for (int e = 0; e < 5; e++) {
    Ent& en = ents[e];
    if (e < 4) snprintf(en.name, sizeof en.name, "tensor_%06d.npy", e);
    else snprintf(en.name, sizeof en.name, "__manifest__.npy");
    size_t nl = strlen(en.name);
    uint8_t hdr[128];
    char shape[24];
    size_t hlen, dlen;
    if (e < 4) {
      const TensorSpec& t = TENSORS[e];
      if (t.cols) snprintf(shape, sizeof shape, "(%d, %d)", t.rows, t.cols);
      else snprintf(shape, sizeof shape, "(%d,)", t.rows);
      hlen = npy_header(hdr, "<f4", shape);
      dlen = (size_t)t.rows * (t.cols ? t.cols : 1) * 4;
    } else {
      snprintf(shape, sizeof shape, "(%u,)", (unsigned)mlen);
      hlen = npy_header(hdr, "|u1", shape);
      dlen = mlen;
    }
    size_t size = hlen + dlen;
    if (o + 30 + nl + size > cap) return 0;
    en.off = (uint32_t)o;
    en.size = (uint32_t)size;
    uint8_t* lh = out + o;
    uint8_t* data = lh + 30 + nl;
    memcpy(data, hdr, hlen);
    if (e < 4) {
      const TensorSpec& t = TENSORS[e];
      int cnt = t.rows * (t.cols ? t.cols : 1);
      for (int i = 0; i < cnt; i++) store_float_le(data + hlen + 4 * i, m.p[t.offset + i]);
    } else {
      memcpy(data + hlen, manifest, mlen);
    }
    en.crc = crc32_update(0, data, size);
    put32(lh, 0x04034B50); put16(lh + 4, 20); put16(lh + 6, 0); put16(lh + 8, 0);
    put16(lh + 10, DOS_TIME); put16(lh + 12, DOS_DATE);
    put32(lh + 14, en.crc); put32(lh + 18, en.size); put32(lh + 22, en.size);
    put16(lh + 26, (uint16_t)nl); put16(lh + 28, 0);
    memcpy(lh + 30, en.name, nl);
    o += 30 + nl + size;
  }
  size_t cd_off = o;
  for (int e = 0; e < 5; e++) {
    const Ent& en = ents[e];
    size_t nl = strlen(en.name);
    if (o + 46 + nl > cap) return 0;
    uint8_t* c = out + o;
    put32(c, 0x02014B50); put16(c + 4, 20); put16(c + 6, 20); put16(c + 8, 0); put16(c + 10, 0);
    put16(c + 12, DOS_TIME); put16(c + 14, DOS_DATE);
    put32(c + 16, en.crc); put32(c + 20, en.size); put32(c + 24, en.size);
    put16(c + 28, (uint16_t)nl); put16(c + 30, 0); put16(c + 32, 0); put16(c + 34, 0); put16(c + 36, 0);
    put32(c + 38, 0); put32(c + 42, en.off);
    memcpy(c + 46, en.name, nl);
    o += 46 + nl;
  }
  size_t cd_size = o - cd_off;
  if (o + 22 > cap) return 0;
  uint8_t* z = out + o;
  put32(z, 0x06054B50); put16(z + 4, 0); put16(z + 6, 0); put16(z + 8, 5); put16(z + 10, 5);
  put32(z + 12, (uint32_t)cd_size); put32(z + 16, (uint32_t)cd_off); put16(z + 20, 0);
  o += 22;
  return o;
}

}  // namespace fl
