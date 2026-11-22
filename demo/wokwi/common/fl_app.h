// fl_app.h - application logic of the FL demo board (included once by sketch.ino).
// Kept in a header so the Arduino .ino preprocessor (auto-prototypes) never touches it.
#pragma once
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <sys/time.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Ed25519.h>  // rweather "Crypto" library
#include <esp_random.h>
#include <time.h>

#include "board_config.h"
#include "board_secrets.h"
#include "board_data.h"
#include "ca_roots.h"
#include "fl_model.h"
#include "fl_sha256.h"
#include "fl_b64.h"
#include "fl_npz.h"
#include "fl_json.h"

// ------------------------------------------------------------------ hardware
#define PIN_SDA 21
#define PIN_SCL 22
#define PIN_LED_GREEN 25
#define PIN_LED_RED 26
#define PIN_BATTERY 34   // potentiometer, 0..4095 -> 0..100 %
#define PIN_OFFLINE 27   // push-button to GND, INPUT_PULLUP, toggles offline
#define PIN_HACKED 14    // slide switch: left = GND (normal), right = open -> pull-up HIGH = HACKED

#define MODEL_VERSION "har-mlp-v1"
#define POLL_MS 5000
#define LOW_BATTERY_PCT 15
#define BODY_CAP 20480      // single request body buffer, reused for both POSTs
#define BODY_OFF 224        // submission JSON starts here; report prefix goes in front
#define HTTP_TIMEOUT_MS 30000

// TLS handshakes + Ed25519 + training run on the loop task: give it 16 KB of stack.
SET_LOOP_TASK_STACK_SIZE(16 * 1024);

static Adafruit_SSD1306 oled(128, 64, &Wire, -1);
static Adafruit_MPU6050 mpu;
static bool oledOk = false, mpuOk = false;

// ------------------------------------------------------------------ state
static fl::Params gModel;  // 9112 B: global model, trained in place
static fl::Params gGrad;   // 9112 B: gradient scratch; also the raw model download buffer
static uint16_t gOrder[FL_N];
static uint16_t gClassCounts[6];
static uint8_t gDevicePub[32];

static WiFiClientSecure gTls;   // persistent connection to the laptop (tunnel)
static HTTPClient gHttp;
static char gBaseUrl[160] = "";  // from the signed beacon
static unsigned long long gBeaconTs = 0;

static char gRoundJson[1024];
static char gResp[2048];      // FinSentinel response body
static char gStatusBuf[512];
static char* gBody = nullptr;  // BODY_CAP bytes, allocated once

static long gLastRound = 0, gCurRound = 0;
static char gState[16] = "boot";
static char gVerdict[22] = "-";
static float gLastLoss = -1.f;
static int gBattery = 100, gRssi = 0;
static bool gOffline = false, gHacked = false;
static int gFailures = 0;
static bool gTlsErrorPending = false;
static char gTlsErrorDetail[96] = "";
static unsigned long gLastPoll = 0, gLastOled = 0;
static int gTrainPct = -1;

// ------------------------------------------------------------------ logging / OLED
#define LOG(...) do { Serial.printf("[%s] ", BOARD_ID); Serial.printf(__VA_ARGS__); Serial.println(); } while (0)

static void logHeap(const char* where) {
  LOG("heap @%s: free %u, largest block %u", where, (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getMaxAllocHeap());
}

static float accelMagnitude() {
  if (!mpuOk) return -1.f;
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  return sqrtf(a.acceleration.x * a.acceleration.x + a.acceleration.y * a.acceleration.y +
               a.acceleration.z * a.acceleration.z);
}

static void drawScreen() {
  if (!oledOk) return;
  bool blink = (millis() / 500) % 2;
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0);
  if (gOffline) {
    oled.setTextSize(2);
    oled.println("OFFLINE");
    oled.setTextSize(1);
    oled.setCursor(0, 20);
    oled.println("button = rejoin");
  } else {
    if (gTrainPct >= 0) oled.printf("R%ld %s %d%%\n", gCurRound, gState, gTrainPct);
    else oled.printf("R%ld %s\n", gCurRound, gState);
    oled.setCursor(0, 11);
    if (gLastLoss >= 0) oled.printf("loss %.3f\n", gLastLoss);
    else oled.println("loss -");
  }
  oled.setCursor(0, 22);
  oled.printf("bat %d%% rssi %d\n", gBattery, gRssi);
  oled.setCursor(0, 33);
  oled.println(gVerdict);
  oled.setCursor(0, 50);
  float am = accelMagnitude();
  if (am >= 0) oled.printf("acc %.1f", am);
  else oled.print("acc --");
  oled.setCursor(56, 50);
  oled.print(BOARD_ID + 6);  // "01".."03"
  if (gHacked && blink) {  // blinking inverted badge (mono OLED: "red" = inverted)
    oled.fillRect(80, 46, 48, 18, SSD1306_WHITE);
    oled.setTextColor(SSD1306_BLACK);
    oled.setCursor(86, 51);
    oled.print("HACKED");
    oled.setTextColor(SSD1306_WHITE);
  }
  oled.display();
}

static void showMessage(const char* l1, const char* l2 = "", const char* l3 = "") {
  LOG("%s %s %s", l1, l2, l3);
  if (!oledOk) return;
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0);  oled.println(l1);
  oled.setCursor(0, 14); oled.println(l2);
  oled.setCursor(0, 28); oled.println(l3);
  oled.display();
}

static void setLeds(bool green, bool red) {
  digitalWrite(PIN_LED_GREEN, green ? HIGH : LOW);
  digitalWrite(PIN_LED_RED, red ? HIGH : LOW);
}

// ------------------------------------------------------------------ inputs
static int readBattery() {
  int raw = analogRead(PIN_BATTERY);
  int pct = (int)((raw * 100L + 2047) / 4095);
  return pct < 0 ? 0 : (pct > 100 ? 100 : pct);
}

static bool gBtnLast = true;
static unsigned long gBtnChange = 0;
static bool gOfflineToggled = false;

static void pollInputs() {
  gBattery = readBattery();
  gHacked = digitalRead(PIN_HACKED) == HIGH;
  if (WiFi.status() == WL_CONNECTED) gRssi = WiFi.RSSI();
  bool b = digitalRead(PIN_OFFLINE) == HIGH;  // HIGH = released
  if (b != gBtnLast && millis() - gBtnChange > 40) {
    gBtnChange = millis();
    gBtnLast = b;
    if (!b) {  // press edge
      gOffline = !gOffline;
      gOfflineToggled = true;
      LOG("offline toggled -> %s", gOffline ? "OFFLINE" : "online");
    }
  }
}

// ------------------------------------------------------------------ HTTP helpers
// Response sink for HTTPClient::writeToStream (handles chunked encoding for us).
class BufStream : public Stream {
 public:
  uint8_t* buf;
  size_t cap, len, total;
  BufStream(uint8_t* b, size_t c) : buf(b), cap(c), len(0), total(0) {}
  size_t write(uint8_t c) override { return write(&c, 1); }
  size_t write(const uint8_t* p, size_t n) override {
    size_t take = (cap > len) ? (cap - len < n ? cap - len : n) : 0;
    if (take) memcpy(buf + len, p, take);
    len += take;
    total += n;
    return n;  // always "accept" so writeToStream keeps draining the socket
  }
  int available() override { return 0; }
  int read() override { return -1; }
  int peek() override { return -1; }
  void flush() {}
  bool overflow() const { return total > len; }
};

static void noteTlsError(WiFiClientSecure& c, const char* host) {
  char err[80] = "";
  int e = c.lastError(err, sizeof err);
  if (e != 0) {
    LOG("TLS refused by %s: %d %s", host, e, err);
    snprintf(gTlsErrorDetail, sizeof gTlsErrorDetail, "%d %s", e, err);
    gTlsErrorPending = true;
    showMessage("TLS refused", host, err);
    strncpy(gVerdict, "TLS refused", sizeof gVerdict);
    setLeds(false, true);
  }
}

// One request on the persistent coordinator connection. Reads the body into out:
// text mode NUL-terminates (stores at most outCap-1 bytes); binary mode stores up
// to outCap bytes. *outLen = bytes stored; gLastRespTotal = bytes the server sent.
// Returns the HTTP code (<0 = transport error).
static size_t gLastRespTotal = 0;
static int httpRequest(const char* path, const char* body, size_t blen, char* out, size_t outCap,
                       size_t* outLen = nullptr, bool text = true) {
  char url[224];
  snprintf(url, sizeof url, "%s%s", gBaseUrl, path);
  unsigned long t0 = millis();
  if (!gHttp.begin(gTls, url)) {
    LOG("http begin failed for %s", url);
    return -100;
  }
  gHttp.setReuse(true);
  gHttp.setConnectTimeout(HTTP_TIMEOUT_MS);
  gHttp.setTimeout(HTTP_TIMEOUT_MS);
  int code;
  if (body) {
    gHttp.addHeader("Content-Type", "application/json");
    code = gHttp.POST((uint8_t*)body, blen);
  } else {
    code = gHttp.GET();
  }
  size_t got = 0;
  if (code > 0) {
    static uint8_t scratch[64];
    BufStream bs(out ? (uint8_t*)out : scratch, out ? (text ? outCap - 1 : outCap) : sizeof scratch);
    gHttp.writeToStream(&bs);
    got = bs.len;
    gLastRespTotal = bs.total;
    if (out && text) out[got] = 0;
    if (bs.overflow()) LOG("response truncated (%u of %u bytes)", (unsigned)bs.len, (unsigned)bs.total);
  } else {
    gLastRespTotal = 0;
    if (out && text) out[0] = 0;
    LOG("%s %s -> transport error %d (%s)", body ? "POST" : "GET", path, code, gHttp.errorToString(code).c_str());
    noteTlsError(gTls, "laptop");
  }
  gHttp.end();
  if (outLen) *outLen = got;
  LOG("%s %s -> %d, %u B, %lu ms", body ? "POST" : "GET", path, code, (unsigned)got, millis() - t0);
  // Transport errors and tunnel gateway errors count as failures (beacon re-read after 2).
  if (code <= 0 || code == 502 || code == 503 || code == 504 || code == 530) gFailures++;
  else gFailures = 0;
  return code;
}

static void postStatus(const char* state, const char* reason = nullptr, const char* detail = nullptr) {
  strncpy(gState, state, sizeof gState - 1);
  gState[sizeof gState - 1] = 0;
  drawScreen();
  if (!gBaseUrl[0]) return;
  fl::JsonBuf j(gStatusBuf, sizeof gStatusBuf);
  j.raw("{\"client_id\":"); j.str(BOARD_ID);
  j.raw(",\"state\":"); j.str(state);
  j.raw(",\"round\":"); j.num(gCurRound);
  j.raw(",\"battery\":"); j.num(gBattery);
  j.raw(",\"signal_dbm\":"); j.num(gRssi);
  j.raw(",\"network\":\"wokwi-guest\",\"online\":"); j.boolean(!gOffline);
  j.raw(",\"hacked\":"); j.boolean(gHacked);
  j.raw(",\"reason\":"); if (reason) j.str(reason); else j.raw("null");
  j.raw(",\"detail\":"); if (detail) j.str(detail, 200); else j.raw("null");
  j.raw("}");
  LOG("status %s%s%s", state, reason ? " / " : "", reason ? reason : "");
  httpRequest("/fl/status", j.p, j.len, nullptr, 0);
}

// ------------------------------------------------------------------ WiFi + time
static void connectWifi() {
  showMessage("WiFi: Wokwi-GUEST", "connecting...", "(can take ~1-2 min)");
  WiFi.mode(WIFI_STA);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    WiFi.disconnect();
    delay(100);
    WiFi.begin("Wokwi-GUEST", "", 6);
    unsigned long a0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - a0 < 40000) delay(250);
    if (WiFi.status() == WL_CONNECTED) break;
    LOG("WiFi not associated after %lu s, retrying", (millis() - t0) / 1000);
    if (millis() - t0 > 120000) {
      showMessage("WiFi failed", "retrying...", "");
      t0 = millis();
    }
  }
  gRssi = WiFi.RSSI();
  LOG("WiFi ok, IP %s, RSSI %d", WiFi.localIP().toString().c_str(), gRssi);
}

static void syncTime() {
  showMessage("WiFi OK", "NTP time sync...");
  configTime(0, 0, "pool.ntp.org", "time.google.com");
  struct tm tm;
  unsigned long t0 = millis();
  while (!getLocalTime(&tm, 1000) || tm.tm_year < (2024 - 1900)) {
    if (millis() - t0 > 30000) {
      LOG("NTP still not synced, re-requesting");
      configTime(0, 0, "pool.ntp.org", "time.google.com");
      t0 = millis();
    }
  }
  char ts[32];
  strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", &tm);
  LOG("UTC time %s", ts);
}

static void utcNow(char out[21]) {
  time_t now = time(nullptr);
  struct tm tm;
  gmtime_r(&now, &tm);
  strftime(out, 21, "%Y-%m-%dT%H:%M:%SZ", &tm);
}

// ------------------------------------------------------------------ beacon
// Streams the ntfy poll response line by line and keeps the 3 newest
// well-formed FLB1 lines; they are then verified newest-first.
class BeaconStream : public Stream {
 public:
  struct Cand { char line[300]; unsigned long long ts; };
  Cand cand[3];
  int ncand = 0;
  char cur[300];
  size_t curLen = 0;
  bool curOverflow = false;
  int lines = 0;

  void endLine() {
    if (curLen && !curOverflow) {
      cur[curLen] = 0;
      lines++;
      char url[160], sig[100];
      unsigned long long ts;
      size_t sl;
      if (fl::parse_beacon(cur, url, sizeof url, &ts, sig, sizeof sig, &sl)) {
        // insert keeping cand[] sorted by ts descending (ties: later line first)
        int pos = 0;
        while (pos < ncand && cand[pos].ts > ts) pos++;
        if (pos < 3) {
          for (int k = (ncand < 3 ? ncand : 2); k > pos; k--) cand[k] = cand[k - 1];
          strcpy(cand[pos].line, cur);
          cand[pos].ts = ts;
          if (ncand < 3) ncand++;
        }
      }
    }
    curLen = 0;
    curOverflow = false;
  }
  size_t write(uint8_t c) override {
    if (c == '\n') endLine();
    else if (c != '\r') {
      if (curLen < sizeof cur - 1) cur[curLen++] = (char)c;
      else curOverflow = true;
    }
    return 1;
  }
  size_t write(const uint8_t* p, size_t n) override {
    for (size_t i = 0; i < n; i++) write(p[i]);
    return n;
  }
  int available() override { return 0; }
  int read() override { return -1; }
  int peek() override { return -1; }
  void flush() {}
};

static bool readBeacon() {
  showMessage("Looking for laptop", "beacon: ntfy.sh", "");
  // Free the coordinator TLS session first (heap), the URL may change anyway.
  gTls.stop();
  static BeaconStream bs;  // ~1.5 KB, keep it off the stack
  bs.ncand = 0; bs.curLen = 0; bs.curOverflow = false; bs.lines = 0;
  bool found = false;
  {
    // Plain HTTP on purpose: every beacon line is Ed25519-signed by the coordinator and is
    // verified below against the pinned COORD_PUBKEY, so TLS adds no integrity here. (ntfy's
    // SHA-384/P-384 certificate chain is too slow to verify on Wokwi's throttled CPU: the
    // server closed the handshake after ~26 s.) A forged or replayed-old beacon is rejected.
    WiFiClient ntfyTcp;
    HTTPClient http;
    char url[160];
    snprintf(url, sizeof url, "http://ntfy.sh/%s/raw?poll=1&since=2h", BEACON_TOPIC);
    http.setConnectTimeout(HTTP_TIMEOUT_MS);
    http.setTimeout(HTTP_TIMEOUT_MS);
    unsigned long t0 = millis();
    int code = -1;
    if (http.begin(ntfyTcp, url)) {
      code = http.GET();
      if (code == 200) http.writeToStream(&bs);
      http.end();
    }
    bs.endLine();
    LOG("beacon GET -> %d, %d lines, %d candidates, %lu ms", code, bs.lines, bs.ncand, millis() - t0);
    if (code <= 0) LOG("beacon fetch failed (HTTP client error %d)", code);
    ntfyTcp.stop();
  }
  for (int i = 0; i < bs.ncand && !found; i++) {
    char url[160], sigb64[100];
    unsigned long long ts;
    size_t sl;
    uint8_t sig[64];
    if (!fl::parse_beacon(bs.cand[i].line, url, sizeof url, &ts, sigb64, sizeof sigb64, &sl)) continue;
    if (fl::b64_decode(sigb64, strlen(sigb64), sig, sizeof sig) != 64) continue;
    unsigned long v0 = millis();
    bool ok = Ed25519::verify(sig, COORD_PUBKEY, bs.cand[i].line, sl);
    LOG("beacon ts=%llu %s -> %s (%lu ms)", ts, url, ok ? "signature OK" : "BAD signature", millis() - v0);
    if (ok) {
      bool changed = strcmp(url, gBaseUrl) != 0;
      strncpy(gBaseUrl, url, sizeof gBaseUrl - 1);
      gBaseUrl[sizeof gBaseUrl - 1] = 0;
      gBeaconTs = ts;
      found = true;
      if (changed) LOG("laptop URL: %s", gBaseUrl);
    }
  }
  if (found) {
    gFailures = 0;
    showMessage("Laptop found", gBaseUrl + 8, "");
    delay(800);
  } else {
    showMessage("No valid beacon", "retry in 10 s", "");
  }
  logHeap("beacon");
  return found;
}

// ------------------------------------------------------------------ training
static void trainHook(int done, int total, void* ctx) {
  int epoch = (int)(intptr_t)ctx;  // 0-based epoch * 1000 + epochs
  int e = epoch / 1000, ne = epoch % 1000;
  gTrainPct = (int)((e * 100L + done * 100L / total) / ne);
  if (millis() - gLastOled > 400) { gLastOled = millis(); drawScreen(); }
  delay(1);  // let the idle task run (watchdog)
}

// ------------------------------------------------------------------ verdict
static void setVerdict(int code, bool* green, bool* red) {
  char st[32] = "";
  *green = *red = false;
  if (code == 200) {
    fl::json_str(gResp, "approval_status", st, sizeof st);
    if (!strcmp(st, "pending_approval") || !strcmp(st, "approved")) {
      snprintf(gVerdict, sizeof gVerdict, "OK 200 %s", !strcmp(st, "approved") ? "APPROVED" : "PENDING");
      *green = true;
    } else if (!strcmp(st, "quarantined")) {
      snprintf(gVerdict, sizeof gVerdict, "QUARANTINED");
      *red = true;
    } else {
      snprintf(gVerdict, sizeof gVerdict, "200 %.14s", st[0] ? st : "?");
      *red = true;
    }
  } else if (code > 0) {
    snprintf(gVerdict, sizeof gVerdict, "BLOCKED %d", code);
    *red = true;
  } else {
    snprintf(gVerdict, sizeof gVerdict, "NO RESPONSE");
    *red = true;
  }
}

// ------------------------------------------------------------------ one round
static void skipRound(const char* state, const char* reason, const char* detail) {
  LOG("round %ld skipped: %s (%s)", gCurRound, reason, detail ? detail : "");
  postStatus(state, reason, detail);
}

static void doRound(const fl::RoundInfo& r) {
  gCurRound = r.round;
  gLastRound = r.round;  // one attempt per round
  gTrainPct = -1;
  LOG("==== round %ld open: epochs %ld lr %.4f batch %ld ====", r.round, r.epochs, r.lr, r.batch_size);
  logHeap("round start");
  pollInputs();

  // 1. controls
  if (gOffline) { skipRound("offline", "offline", "offline button"); return; }
  if (gBattery < LOW_BATTERY_PCT) {
    char d[32]; snprintf(d, sizeof d, "battery %d%%", gBattery);
    strncpy(gVerdict, "SKIP low battery", sizeof gVerdict);
    skipRound("skipped", "low_battery", d);
    return;
  }

  // 2. download + verify the signed global model
  postStatus("downloading");
  char path[32];
  snprintf(path, sizeof path, "/fl/model/%ld", r.round);
  size_t got = 0;
  uint8_t* raw = (uint8_t*)gGrad.p;  // 9112 B scratch
  int code = httpRequest(path, nullptr, 0, (char*)raw, fl::MODEL_BYTES, &got, /*text=*/false);
  if (code != 200 || got != fl::MODEL_BYTES || gLastRespTotal != fl::MODEL_BYTES) {
    char d[48]; snprintf(d, sizeof d, "model download http %d, %u B", code, (unsigned)gLastRespTotal);
    skipRound("skipped", "weak_signal", d);
    return;
  }
  char hex[65];
  fl::sha256_hex(raw, fl::MODEL_BYTES, hex);
  bool shaOk = strcmp(hex, r.model_sha256) == 0;
  bool verOk = strcmp(r.model_version, MODEL_VERSION) == 0;
  uint8_t sig[64];
  bool sigOk = false;
  if (shaOk && verOk && fl::b64_decode(r.model_sig, strlen(r.model_sig), sig, sizeof sig) == 64) {
    char msg[128];
    int ml = snprintf(msg, sizeof msg, "fl-model|%s|%ld|%s", MODEL_VERSION, r.round, r.model_sha256);
    unsigned long v0 = millis();
    sigOk = Ed25519::verify(sig, COORD_PUBKEY, msg, (size_t)ml);
    LOG("model signature %s (%lu ms)", sigOk ? "OK" : "BAD", millis() - v0);
  }
  if (!shaOk || !verOk || !sigOk) {
    const char* why = !verOk ? "unexpected model_version" : (!shaOk ? "sha256 mismatch" : "bad coordinator signature");
    LOG("MODEL REJECTED: %s (got sha %s)", why, hex);
    strncpy(gVerdict, "MODEL REJECTED", sizeof gVerdict);
    setLeds(false, true);
    skipRound("model_rejected", "model_rejected", why);
    return;
  }
  fl::load_bytes(gModel, raw, fl::MODEL_BYTES);
  LOG("model verified: sha %s", hex);

  // 3. train
  setLeds(false, false);
  char detail[48];
  snprintf(detail, sizeof detail, "%ld epoch(s), %d samples", r.epochs, (int)FL_N);
  postStatus("training", nullptr, detail);
  float losses[20];
  unsigned long t0 = millis();
  for (long e = 0; e < r.epochs; e++) {
    fl::shuffle_order(gOrder, FL_N, esp_random);
    losses[e] = fl::sgd_epoch(gModel, gGrad, &FL_X[0][0], FL_Y, FL_N, 1.0f / 127.0f, r.lr, (int)r.batch_size,
                              gOrder, trainHook, (void*)(intptr_t)(e * 1000 + r.epochs));
    gLastLoss = losses[e];
    LOG("epoch %ld/%ld loss %.4f", e + 1, r.epochs, losses[e]);
  }
  float seconds = (millis() - t0) / 1000.f;  // simulated time
  gTrainPct = -1;
  pollInputs();
  bool hacked = gHacked;
  if (hacked) {
    fl::scale_params(gModel, 8.0f);
    LOG("HACKED switch ON: weights x8 (poisoned update)");
  }

  // 4. payload + signed envelope
  postStatus("signing");
  logHeap("before npz");
  uint8_t* npz = (uint8_t*)malloc(fl::NPZ_SIZE + 64);
  if (!npz) { skipRound("error", "other", "out of memory (npz)"); return; }
  size_t npzLen = fl::build_npz(gModel, npz, fl::NPZ_SIZE + 64);
  char payloadSha[65];
  fl::sha256_hex(npz, npzLen, payloadSha);
  char createdAt[21], updateId[40];
  utcNow(createdAt);
  snprintf(updateId, sizeof updateId, "%s-r%04ld-%08lx", BOARD_ID, r.round, (unsigned long)esp_random());
  fl::Envelope env{BOARD_ID, createdAt, MODEL_VERSION, payloadSha, (unsigned long)npzLen, r.round, (long)FL_N,
                   SIGNING_KEY_ID, updateId};
  char canon[512];
  size_t cl = fl::canonical_envelope(env, canon, sizeof canon);
  uint8_t esig[64];
  unsigned long s0 = millis();
  Ed25519::sign(esig, DEVICE_SEED, gDevicePub, canon, cl);
  LOG("envelope signed (%lu ms): %s", millis() - s0, canon);
  char esigB64[96];
  fl::b64_encode(esig, 64, esigB64);

  // Submission JSON at gBody + BODY_OFF: {"envelope":{...,"signature":"..."},"payload_base64":"..."}
  char* sub = gBody + BODY_OFF;
  size_t cap = BODY_CAP - BODY_OFF;
  size_t n = 0;
  n += snprintf(sub + n, cap - n, "{\"envelope\":");
  memcpy(sub + n, canon, cl - 1); n += cl - 1;  // canonical without the closing '}'
  n += snprintf(sub + n, cap - n, ",\"signature\":\"%s\"},\"payload_base64\":\"", esigB64);
  n += fl::b64_encode(npz, npzLen, sub + n);
  free(npz);
  npz = nullptr;
  sub[n++] = '"';
  sub[n++] = '}';
  sub[n] = 0;
  size_t subLen = n;
  logHeap("body built");

  // 5. submit to FinSentinel
  postStatus("sending");
  size_t respLen = 0;
  int fsCode = httpRequest("/api/v1/updates", sub, subLen, gResp, sizeof gResp, &respLen);
  bool respComplete = gLastRespTotal == respLen;
  bool green, red;
  setVerdict(fsCode, &green, &red);
  LOG("FinSentinel -> %d: %.300s", fsCode, gResp);

  // 6. report: prefix written right in front of the submission, suffix after it.
  char prefix[96];
  int pl = snprintf(prefix, sizeof prefix, "{\"client_id\":\"%s\",\"round\":%ld,\"submission\":", BOARD_ID, r.round);
  char* rep = sub - pl;
  memcpy(rep, prefix, pl);
  fl::JsonBuf j(sub + subLen, cap - subLen);
  j.raw(",\"finsentinel\":{\"http_status\":"); j.num(fsCode);
  j.raw(",\"body\":");
  bool isObj = respComplete && respLen > 1 && gResp[0] == '{' && gResp[respLen - 1] == '}';
  if (isObj) j.raw(gResp, respLen);
  else { j.raw("{\"text\":"); j.str(gResp, 600); j.raw("}"); }
  j.raw("},\"train\":{\"epochs\":"); j.num(r.epochs);
  j.raw(",\"losses\":[");
  for (long e = 0; e < r.epochs; e++) { if (e) j.raw(","); j.flt(losses[e], 5); }
  j.raw("],\"samples\":"); j.num(FL_N);
  j.raw(",\"class_counts\":[");
  for (int c = 0; c < 6; c++) { if (c) j.raw(","); j.num(gClassCounts[c]); }
  j.raw("],\"seconds\":"); j.flt(seconds, 2);
  j.raw("},\"battery\":"); j.num(gBattery);
  j.raw(",\"signal_dbm\":"); j.num(gRssi);
  j.raw(",\"hacked\":"); j.boolean(hacked);
  j.raw("}");
  if (j.overflow) LOG("WARNING: report truncated (buffer)");
  size_t repLen = (size_t)pl + subLen + j.len;
  httpRequest("/fl/report", rep, repLen, nullptr, 0);

  setLeds(green, red);
  postStatus("done", nullptr, gVerdict);
  logHeap("round end");
}

// ------------------------------------------------------------------ setup / loop
static void app_setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println();
  LOG("FL board %s (%s), subject %d, %d samples", BOARD_ID, BOARD_LABEL, BOARD_SUBJECT, (int)FL_N);
  LOG("arduino-esp32 %s, cpu %lu MHz", ESP_ARDUINO_VERSION_STR, (unsigned long)getCpuFrequencyMhz());

  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);
  pinMode(PIN_OFFLINE, INPUT_PULLUP);
  pinMode(PIN_HACKED, INPUT_PULLUP);
  setLeds(false, false);
  analogReadResolution(12);

  Wire.begin(PIN_SDA, PIN_SCL);
  oledOk = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (!oledOk) LOG("OLED init failed");
  mpuOk = mpu.begin(0x68, &Wire);
  if (!mpuOk) LOG("MPU6050 init failed");
  char l1[24];
  snprintf(l1, sizeof l1, "FL board %s", BOARD_ID);
  showMessage(l1, "Wokwi-simulated", BOARD_LABEL);
  delay(1200);

  for (int s = 0; s < FL_N; s++) if (FL_Y[s] < 6) gClassCounts[FL_Y[s]]++;
  Ed25519::derivePublicKey(gDevicePub, DEVICE_SEED);
  char pubHex[65];
  fl::to_hex(gDevicePub, 32, pubHex);
  LOG("device public key %s", pubHex);

  gBody = (char*)malloc(BODY_CAP);
  if (!gBody) LOG("FATAL: cannot allocate %d B body buffer", BODY_CAP);
  logHeap("setup");

  connectWifi();
  syncTime();
  gTls.setCACert(CA_ROOTS);  // public roots only; never setInsecure()
  gHttp.setReuse(true);
  while (!readBeacon()) delay(10000);
  postStatus("online");
  logHeap("ready");
}

static void app_loop() {
  pollInputs();
  if (millis() - gLastOled > 500) { gLastOled = millis(); drawScreen(); }

  if (gOfflineToggled) {
    gOfflineToggled = false;
    // Telemetry about the toggle itself is still sent (it is the "offline" event).
    postStatus(gOffline ? "offline" : "online", gOffline ? "offline" : nullptr, "button");
    if (!gOffline) gLastPoll = 0;
  }
  if (gOffline) { delay(50); return; }

  if (WiFi.status() != WL_CONNECTED) {
    LOG("WiFi lost");
    connectWifi();
  }
  if (millis() - gLastPoll < POLL_MS) { delay(50); return; }
  gLastPoll = millis();

  if (!gBaseUrl[0] || gFailures >= 2) {
    if (gFailures >= 2) LOG("%d consecutive failures -> re-reading beacon", gFailures);
    if (!readBeacon()) { delay(5000); return; }  // retry in ~10 s
  }

  size_t len = 0;
  int code = httpRequest("/fl/round", nullptr, 0, gRoundJson, sizeof gRoundJson, &len);
  if (code != 200) return;
  if (gTlsErrorPending) {  // connection works again: report the earlier TLS refusal
    gTlsErrorPending = false;
    postStatus("error", "tls_error", gTlsErrorDetail);
  }
  fl::RoundInfo r;
  if (!fl::parse_round(gRoundJson, r)) { LOG("bad /fl/round JSON: %.200s", gRoundJson); return; }
  // Wokwi runs slower than real time, so the clock set once by NTP drifts behind and
  // FinSentinel rejects stale envelopes (403 "exceeds the permitted age"). Re-sync from the
  // coordinator's server_time, which arrives over the certificate-checked TLS connection.
  if (r.server_time > 1.7e9) {
    time_t now = time(nullptr);
    long long drift = (long long)r.server_time - (long long)now;
    if (drift > 2 || drift < -2) {
      struct timeval tv = {(time_t)r.server_time, 0};
      settimeofday(&tv, nullptr);
      LOG("clock re-synced from coordinator (was %+lld s off)", drift);
    }
  }
  if (r.round != gCurRound && strcmp(r.state, "open") != 0) {
    gCurRound = r.round;
  }
  if (strcmp(r.state, "open") == 0 && r.round != gLastRound) {
    doRound(r);
    gLastPoll = millis();
  } else if (strcmp(gState, "done") != 0 && strcmp(gState, "idle") != 0 && strcmp(r.state, "open") != 0) {
    postStatus("idle", nullptr, r.state);
  }
}
