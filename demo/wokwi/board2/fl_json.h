// fl_json.h - canonical envelope string and tiny JSON helpers. Host-testable.
//
// canonical_envelope() reproduces fldemo.wire.canonical_envelope(): keys sorted,
// separators (",", ":"), integers bare. All string fields are plain ASCII
// (ids, ISO timestamp, hex digest) so no escaping is required there.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace fl {

struct Envelope {
  const char* client_id;
  const char* created_at;     // YYYY-MM-DDTHH:MM:SSZ
  const char* model_version;  // har-mlp-v1
  const char* payload_sha256; // 64 lowercase hex
  unsigned long payload_size_bytes;
  long round_number;
  long sample_count;
  const char* signing_key_id;
  const char* update_id;
};

// Returns length written (excluding NUL), or 0 on overflow.
inline size_t canonical_envelope(const Envelope& e, char* out, size_t cap) {
  int n = snprintf(out, cap,
                   "{\"client_id\":\"%s\",\"created_at\":\"%s\",\"model_version\":\"%s\","
                   "\"payload_sha256\":\"%s\",\"payload_size_bytes\":%lu,\"round_number\":%ld,"
                   "\"sample_count\":%ld,\"signing_key_id\":\"%s\",\"update_id\":\"%s\"}",
                   e.client_id, e.created_at, e.model_version, e.payload_sha256, e.payload_size_bytes,
                   e.round_number, e.sample_count, e.signing_key_id, e.update_id);
  if (n < 0 || (size_t)n >= cap) return 0;
  return (size_t)n;
}

// ---------------------------------------------------------------- bounded writer
struct JsonBuf {
  char* p;
  size_t cap, len;
  bool overflow;
  JsonBuf(char* buf, size_t c) : p(buf), cap(c), len(0), overflow(false) { if (cap) p[0] = 0; }
  void raw(const char* s, size_t n) {
    if (len + n + 1 > cap) { overflow = true; return; }
    memcpy(p + len, s, n); len += n; p[len] = 0;
  }
  void raw(const char* s) { raw(s, strlen(s)); }
  void fmt(const char* f, long v) { char t[32]; int n = snprintf(t, sizeof t, f, v); raw(t, (size_t)n); }
  void num(long v) { fmt("%ld", v); }
  void flt(float v, int decimals = 4) {
    char t[32];
    int n = snprintf(t, sizeof t, "%.*f", decimals, (double)v);
    raw(t, (size_t)n);
  }
  // JSON string literal with escaping (control chars -> \u00XX), at most maxlen input bytes.
  void str(const char* s, size_t maxlen = (size_t)-1) {
    raw("\"", 1);
    for (size_t i = 0; s[i] && i < maxlen; i++) {
      unsigned char c = (unsigned char)s[i];
      if (c == '"') raw("\\\"", 2);
      else if (c == '\\') raw("\\\\", 2);
      else if (c == '\n') raw("\\n", 2);
      else if (c == '\r') raw("\\r", 2);
      else if (c == '\t') raw("\\t", 2);
      else if (c < 0x20) { char t[8]; snprintf(t, sizeof t, "\\u%04x", c); raw(t, 6); }
      else raw((const char*)&c, 1);
    }
    raw("\"", 1);
  }
  void boolean(bool b) { raw(b ? "true" : "false"); }
};

// ---------------------------------------------------------------- extraction
// Minimal lookup of a top-level-ish key in a flat JSON object: finds "key"
// followed by optional whitespace and ':'. Good enough for /fl/round and
// FinSentinel responses (keys are unique there).
inline const char* json_find(const char* js, const char* key) {
  size_t kl = strlen(key);
  const char* p = js;
  while ((p = strchr(p, '"')) != nullptr) {
    if (strncmp(p + 1, key, kl) == 0 && p[1 + kl] == '"') {
      const char* q = p + 2 + kl;
      while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
      if (*q == ':') {
        q++;
        while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
        return q;
      }
    }
    // skip to the end of this string literal
    p++;
    while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; p++; }
    if (!*p) return nullptr;
    p++;
  }
  return nullptr;
}

// Copies a string value (unescaping \" and \\ only). Returns false if missing / not a string.
inline bool json_str(const char* js, const char* key, char* out, size_t cap) {
  const char* v = json_find(js, key);
  if (!v || *v != '"' || cap == 0) return false;
  v++;
  size_t o = 0;
  while (*v && *v != '"') {
    char c = *v++;
    if (c == '\\' && *v) c = *v++;
    if (o + 1 < cap) out[o++] = c;
  }
  out[o] = 0;
  return *v == '"';
}

inline bool json_num(const char* js, const char* key, double* out) {
  const char* v = json_find(js, key);
  if (!v) return false;
  char* end = nullptr;
  double d = strtod(v, &end);
  if (end == v) return false;
  *out = d;
  return true;
}

inline bool json_long(const char* js, const char* key, long* out) {
  double d;
  if (!json_num(js, key, &d)) return false;
  *out = (long)d;
  return true;
}

struct RoundInfo {
  long round;
  char state[16];
  char model_version[24];
  char model_sha256[65];
  char model_sig[100];
  double deadline, server_time;
  long epochs;
  float lr;
  long batch_size;
};

inline bool parse_round(const char* js, RoundInfo& r) {
  memset(&r, 0, sizeof r);
  double lr = 0.05;
  if (!json_long(js, "round", &r.round)) return false;
  if (!json_str(js, "state", r.state, sizeof r.state)) return false;
  json_str(js, "model_version", r.model_version, sizeof r.model_version);
  json_str(js, "model_sha256", r.model_sha256, sizeof r.model_sha256);
  json_str(js, "model_sig", r.model_sig, sizeof r.model_sig);
  json_num(js, "deadline", &r.deadline);
  json_num(js, "server_time", &r.server_time);
  if (!json_long(js, "epochs", &r.epochs)) r.epochs = 1;
  json_num(js, "lr", &lr);
  r.lr = (float)lr;
  if (!json_long(js, "batch_size", &r.batch_size)) r.batch_size = 32;
  if (r.epochs < 1) r.epochs = 1;
  if (r.epochs > 20) r.epochs = 20;
  if (r.batch_size < 1) r.batch_size = 1;
  return true;
}

// Beacon line "FLB1|<url>|<unix seconds>|<base64 sig>". On success url/sig are
// NUL-terminated copies, signed_len is the length of "FLB1|<url>|<ts>".
inline bool parse_beacon(const char* line, char* url, size_t url_cap, unsigned long long* ts,
                         char* sig, size_t sig_cap, size_t* signed_len) {
  if (strncmp(line, "FLB1|", 5) != 0) return false;
  const char* u = line + 5;
  const char* p1 = strchr(u, '|');
  if (!p1) return false;
  const char* p2 = strchr(p1 + 1, '|');
  if (!p2) return false;
  size_t ul = (size_t)(p1 - u);
  if (ul == 0 || ul >= url_cap || strncmp(u, "https://", 8) != 0) return false;
  memcpy(url, u, ul); url[ul] = 0;
  char* end = nullptr;
  *ts = strtoull(p1 + 1, &end, 10);
  if (end != p2) return false;
  const char* s = p2 + 1;
  size_t sl = 0;
  while (s[sl] && s[sl] != '\r' && s[sl] != '\n' && s[sl] != ' ') sl++;
  if (sl == 0 || sl >= sig_cap) return false;
  memcpy(sig, s, sl); sig[sl] = 0;
  *signed_len = (size_t)(p2 - line);
  return true;
}

}  // namespace fl
