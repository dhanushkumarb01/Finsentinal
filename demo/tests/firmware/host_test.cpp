// Host test driver for the pure firmware headers in demo/wokwi/common/.
// Driven by demo/tests/test_firmware_host.py; data passes through files.
//
//   host_test forward  <model.bin> <x_i8.bin> <n> <out_logits.f32>
//   host_test epoch    <model.bin> <x_i8.bin> <y_u8.bin> <n> <lr> <batch> <order_u16.bin|-> <out_model.bin>
//                      (prints "loss <value>")
//   host_test npz      <model.bin> <out.npz>
//   host_test envelope <client_id> <created_at> <model_version> <sha> <size> <round> <samples> <key_id> <update_id>
//   host_test sha256   <file>            (prints hex)
//   host_test b64enc   <file>            (prints base64)
//   host_test b64dec   <in.txt> <out.bin>
//   host_test roundtrip <model.bin> <out.bin>
//   host_test round    <json file>       (prints parsed fields)
//   host_test beacon   <line file>       (prints parsed fields)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#include "fl_model.h"
#include "fl_sha256.h"
#include "fl_b64.h"
#include "fl_npz.h"
#include "fl_json.h"

static std::vector<uint8_t> read_file(const char* path) {
  std::vector<uint8_t> v;
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
  uint8_t buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
  fclose(f);
  return v;
}

static void write_file(const char* path, const void* data, size_t n) {
  FILE* f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
  fwrite(data, 1, n, f);
  fclose(f);
}

static fl::Params g_model, g_grad;

static void load_model(const char* path) {
  std::vector<uint8_t> b = read_file(path);
  if (!fl::load_bytes(g_model, b.data(), b.size())) { fprintf(stderr, "bad model size %zu\n", b.size()); exit(3); }
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: see source\n"); return 1; }
  std::string cmd = argv[1];
  const float scale = 1.0f / 127.0f;

  if (cmd == "forward" && argc == 6) {
    load_model(argv[2]);
    std::vector<uint8_t> x = read_file(argv[3]);
    int n = atoi(argv[4]);
    std::vector<float> out((size_t)n * fl::NC);
    float xf[fl::NF], h[fl::NH];
    for (int s = 0; s < n; s++) {
      fl::load_row((const int8_t*)x.data(), s, scale, xf);
      fl::forward_f(g_model, xf, nullptr, h, &out[(size_t)s * fl::NC]);
    }
    write_file(argv[5], out.data(), out.size() * 4);
    return 0;
  }
  if (cmd == "epoch" && argc == 10) {
    load_model(argv[2]);
    std::vector<uint8_t> x = read_file(argv[3]);
    std::vector<uint8_t> y = read_file(argv[4]);
    int n = atoi(argv[5]);
    float lr = (float)atof(argv[6]);
    int batch = atoi(argv[7]);
    std::vector<uint8_t> ord;
    const uint16_t* order = nullptr;
    if (strcmp(argv[8], "-") != 0) { ord = read_file(argv[8]); order = (const uint16_t*)ord.data(); }
    float loss = fl::sgd_epoch(g_model, g_grad, (const int8_t*)x.data(), y.data(), n, scale, lr, batch, order);
    uint8_t out[fl::MODEL_BYTES];
    fl::to_bytes(g_model, out);
    write_file(argv[9], out, sizeof out);
    printf("loss %.9g\n", (double)loss);
    return 0;
  }
  if (cmd == "npz" && argc == 4) {
    load_model(argv[2]);
    static uint8_t buf[16384];
    size_t n = fl::build_npz(g_model, buf, sizeof buf);
    if (!n) { fprintf(stderr, "npz overflow\n"); return 4; }
    write_file(argv[3], buf, n);
    printf("size %zu\n", n);
    return 0;
  }
  if (cmd == "envelope" && argc == 11) {
    fl::Envelope e{argv[2], argv[3], argv[4], argv[5], strtoul(argv[6], nullptr, 10),
                   atol(argv[7]), atol(argv[8]), argv[9], argv[10]};
    char out[1024];
    if (!fl::canonical_envelope(e, out, sizeof out)) return 5;
    fputs(out, stdout);
    return 0;
  }
  if (cmd == "sha256" && argc == 3) {
    std::vector<uint8_t> d = read_file(argv[2]);
    char hex[65];
    fl::sha256_hex(d.data(), d.size(), hex);
    puts(hex);
    return 0;
  }
  if (cmd == "b64enc" && argc == 3) {
    std::vector<uint8_t> d = read_file(argv[2]);
    std::vector<char> out(fl::b64_len(d.size()) + 1);
    fl::b64_encode(d.data(), d.size(), out.data());
    fputs(out.data(), stdout);
    return 0;
  }
  if (cmd == "b64dec" && argc == 4) {
    std::vector<uint8_t> d = read_file(argv[2]);
    std::vector<uint8_t> out(d.size() + 3);
    long n = fl::b64_decode((const char*)d.data(), d.size(), out.data(), out.size());
    if (n < 0) { puts("invalid"); return 0; }
    write_file(argv[3], out.data(), (size_t)n);
    printf("ok %ld\n", n);
    return 0;
  }
  if (cmd == "roundtrip" && argc == 4) {
    load_model(argv[2]);
    uint8_t out[fl::MODEL_BYTES];
    fl::to_bytes(g_model, out);
    write_file(argv[3], out, sizeof out);
    return 0;
  }
  if (cmd == "round" && argc == 3) {
    std::vector<uint8_t> d = read_file(argv[2]);
    d.push_back(0);
    fl::RoundInfo r;
    if (!fl::parse_round((const char*)d.data(), r)) { puts("parse_error"); return 0; }
    printf("round=%ld state=%s version=%s sha=%s sig=%s epochs=%ld lr=%.4f batch=%ld deadline=%.1f\n",
           r.round, r.state, r.model_version, r.model_sha256, r.model_sig, r.epochs, (double)r.lr,
           r.batch_size, r.deadline);
    return 0;
  }
  if (cmd == "beacon" && argc == 3) {
    std::vector<uint8_t> d = read_file(argv[2]);
    d.push_back(0);
    char url[160], sig[100];
    unsigned long long ts = 0;
    size_t sl = 0;
    if (!fl::parse_beacon((const char*)d.data(), url, sizeof url, &ts, sig, sizeof sig, &sl)) {
      puts("invalid");
      return 0;
    }
    printf("url=%s ts=%llu sig=%s signed_len=%zu\n", url, ts, sig, sl);
    return 0;
  }
  fprintf(stderr, "bad command/args\n");
  return 1;
}
