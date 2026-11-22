#!/usr/bin/env bash
# Wazuh for the FL demo — run INSIDE WSL Ubuntu.
#
#   bash /mnt/c/Users/Dhanush/drift-causal-fl/demo/wazuh/wazuh.sh setup    # once (asks for your sudo password)
#   bash .../demo/wazuh/wazuh.sh up        # start (run_demo.py --wazuh calls this for you)
#   bash .../demo/wazuh/wazuh.sh status | down | logtest | decisions
#
# Uses FinSentinel's Wazuh deployment UNMODIFIED (finsentinel/deploy/wazuh: compose file, certs,
# rules, setup-wsl.sh) and only overlays the demo's decision rules + active response.
set -euo pipefail

DEMO_WAZUH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DEMO_WAZUH/../.." && pwd)"
FS_WAZUH_SRC="$REPO/finsentinel/deploy/wazuh"   # read-only source (CRLF on this Windows checkout)
ENV_DIR="$HOME/finsentinel-wazuh"
FS_WAZUH="$ENV_DIR/deploy"                        # Linux-filesystem mirror we actually run
ENV_FILE="$ENV_DIR/.env"
SIEM_DIR="$REPO/demo/.state/siem"
DECISIONS_DIR="$REPO/demo/.state/wazuh"
LINUX_DIR="$ENV_DIR/fl-demo"

docker_cmd() { if docker info >/dev/null 2>&1; then docker "$@"; else sudo docker "$@"; fi; }
compose() {
  FLDEMO_LINUX_DIR="$LINUX_DIR" FLDEMO_DECISIONS_DIR="$DECISIONS_DIR" \
    docker_cmd compose --project-directory "$FS_WAZUH" --env-file "$ENV_FILE" \
      -f "$FS_WAZUH/docker-compose.yml" -f "$DEMO_WAZUH/docker-compose.demo.yml" "$@"
}

mirror_finsentinel_deploy() {
  # Copy FinSentinel's Wazuh deployment to the Linux filesystem and normalise line endings there.
  # The repo copy is never modified; generated certs stay in the mirror (outside the repo).
  mkdir -p "$FS_WAZUH"
  # certs made by the root-run generator container: make them ours before touching the tree
  if [ -d "$FS_WAZUH/config/wazuh_indexer_ssl_certs" ]; then
    sudo chown -R "$USER:$USER" "$FS_WAZUH/config/wazuh_indexer_ssl_certs"
  fi
  cp -r "$FS_WAZUH_SRC/." "$FS_WAZUH/"
  find "$FS_WAZUH" -type f \( -name "*.sh" -o -name "*.yml" -o -name "*.xml" -o -name "*.conf" -o -name ".env.example" \) -exec sed -i 's/\r$//' {} +
}

point_logs_at_demo() {
  mkdir -p "$ENV_DIR" "$SIEM_DIR" "$DECISIONS_DIR"
  touch "$SIEM_DIR/finsentinel-siem.jsonl" "$SIEM_DIR/fl-coordinator.jsonl"
  if [ ! -f "$ENV_FILE" ]; then
    cp "$FS_WAZUH/.env.example" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
  fi
  sed -i 's/\r$//' "$ENV_FILE"   # a CRLF .env breaks the API password (found 2026-11-22)
  if grep -q '^FINSENTINEL_LOGS_DIR=' "$ENV_FILE"; then
    sed -i "s|^FINSENTINEL_LOGS_DIR=.*|FINSENTINEL_LOGS_DIR=$SIEM_DIR|" "$ENV_FILE"
  else
    echo "FINSENTINEL_LOGS_DIR=$SIEM_DIR" >> "$ENV_FILE"
  fi
}

stage_demo_files() {
  # Linux-filesystem copies with sane permissions (drvfs files look 0777).
  mkdir -p "$LINUX_DIR"
  src_conf="$ENV_DIR/config/wazuh_cluster/wazuh_manager.conf"
  [ -f "$src_conf" ] || src_conf="$FS_WAZUH/config/wazuh_cluster/wazuh_manager.conf"
  # Files may have been chowned to root:wazuh inside the container (bind mounts): stage with sudo.
  tmp="$(mktemp -d)"
  cat "$src_conf" "$DEMO_WAZUH/manager_overlay.conf" | tr -d '\r' > "$tmp/ossec.conf"
  tr -d '\r' < "$DEMO_WAZUH/fl_decisions.xml" > "$tmp/fl_decisions.xml"
  tr -d '\r' < "$DEMO_WAZUH/fl-decision" > "$tmp/fl-decision"
  sudo install -m 0644 "$tmp/ossec.conf" "$LINUX_DIR/ossec.conf"
  sudo install -m 0644 "$tmp/fl_decisions.xml" "$LINUX_DIR/fl_decisions.xml"
  sudo install -m 0755 "$tmp/fl-decision" "$LINUX_DIR/fl-decision"
  rm -rf "$tmp"
}

api_ok() {
  local user pass
  user="$(grep -E '^API_USERNAME=' "$ENV_FILE" | cut -d= -f2-)"; user="${user:-wazuh-wui}"
  pass="$(grep -E '^API_PASSWORD=' "$ENV_FILE" | cut -d= -f2-)"
  curl -s -k -m 5 -X POST -u "$user:$pass" https://127.0.0.1:55000/security/user/authenticate | grep -q '"token"'
}

case "${1:-status}" in
  setup)
    mirror_finsentinel_deploy
    point_logs_at_demo
    echo "[fl-demo] Running FinSentinel's setup-wsl.sh (Docker Engine, certs, stack) ..."
    # The cert generator runs as root in a container; on the Linux filesystem its output is then
    # root-only (on FinSentinel's original drvfs location everything was readable). Give the
    # certs back to this user before/after FinSentinel's script uses them.
    certs="$FS_WAZUH/config/wazuh_indexer_ssl_certs"
    [ -d "$certs" ] && sudo chown -R "$USER:$USER" "$certs"
    if ! bash "$FS_WAZUH/setup-wsl.sh"; then
      [ -d "$certs" ] && sudo chown -R "$USER:$USER" "$certs"
      echo "[fl-demo] retrying FinSentinel's setup-wsl.sh after fixing cert ownership ..."
      bash "$FS_WAZUH/setup-wsl.sh"
    fi
    stage_demo_files
    compose up -d
    echo "[fl-demo] Setup done. The dashboard needs a few more minutes: https://localhost"
    ;;
  up)
    mirror_finsentinel_deploy
    point_logs_at_demo
    stage_demo_files
    if ! command -v docker >/dev/null 2>&1; then echo "Docker not installed — run: $0 setup" >&2; exit 2; fi
    # First start may report the manager unhealthy (missing ar.conf) and skip the dashboard: tolerate
    # it, bootstrap, restart the manager, then bring the dashboard up once the API answers.
    compose up -d || echo "[fl-demo] first start incomplete — bootstrapping manager"
    docker_cmd exec wazuh-manager bash -c \
      'mkdir -p /var/ossec/etc/shared/default && touch /var/ossec/etc/shared/ar.conf && chown -R root:wazuh /var/ossec/etc/shared && chmod 770 /var/ossec/etc/shared && chmod 660 /var/ossec/etc/shared/ar.conf && chown root:wazuh /var/ossec/active-response/bin/fl-decision 2>/dev/null; true' || true
    # analysisd needs ar.conf at start; it only exists after a first successful start (chicken-and-egg,
    # same fix as FinSentinel's setup-wsl.sh step 8). Restart only if the API is not up yet.
    api_ok || docker_cmd restart wazuh-manager >/dev/null
    echo -n "[fl-demo] waiting for the Wazuh manager API"
    for _ in $(seq 1 120); do
      if api_ok; then
        echo " — up"
        compose up -d wazuh.dashboard >/dev/null 2>&1 || echo "[fl-demo] dashboard still starting (it can take several minutes)"
        exit 0
      fi
      echo -n "."; sleep 5
    done
    echo " — not up after 10 min (see: $0 status)" >&2; exit 1
    ;;
  status)
    compose ps || true
    if api_ok; then echo "manager API: UP"; else echo "manager API: DOWN"; fi
    ;;
  down)
    compose stop
    ;;
  decisions)
    tail -n 20 "$DECISIONS_DIR/decisions.jsonl" 2>/dev/null || echo "(no decisions yet)"
    ;;
  logtest)
    # Feeds sample events through the live ruleset and shows which rule fires. Real events always carry
    # a timestamp: Wazuh's JSON parent rule 86600 (FinSentinel's rules chain off it) requires one.
    TS='"timestamp":"2026-11-22T17:00:00"'
    for ev in \
      "{\"source\":\"finsentinel\",$TS,\"event_type\":\"security.anomaly_clean\",\"severity\":\"info\",\"client_id\":\"phone-01\",\"update_id\":\"phone-01-r0001-aaaa\"}" \
      "{\"source\":\"finsentinel\",$TS,\"event_type\":\"security.anomaly_detected\",\"severity\":\"high\",\"client_id\":\"phone-03\",\"update_id\":\"phone-03-r0002-bbbb\"}" \
      "{\"source\":\"finsentinel\",$TS,\"event_type\":\"security.anomaly_detected\",\"severity\":\"critical\",\"client_id\":\"phone-03\",\"update_id\":\"phone-03-r0003-cccc\"}" \
      "{\"source\":\"finsentinel\",$TS,\"event_type\":\"security.replay_detected\",\"severity\":\"high\",\"client_id\":\"phone-04\",\"update_id\":\"phone-04-r0003-dddd\"}" \
      "{\"source\":\"finsentinel\",$TS,\"event_type\":\"security.signature_verification_failed\",\"severity\":\"high\",\"client_id\":\"phone-02\",\"update_id\":\"x\"}" \
      "{\"source\":\"finsentinel\",$TS,\"event_type\":\"security.payload_integrity_failed\",\"severity\":\"high\",\"client_id\":\"phone-04\",\"update_id\":\"y\"}" \
      "{\"source\":\"fl-coordinator\",$TS,\"event_type\":\"fl.drift_dropout\",\"severity\":\"low\",\"client_id\":\"phone-05\"}" \
      "{\"source\":\"fl-coordinator\",$TS,\"event_type\":\"fl.mitm_refused\",\"severity\":\"high\",\"client_id\":\"\"}"; do
      out="$(echo "$ev" | docker_cmd exec -i wazuh-manager /var/ossec/bin/wazuh-logtest 2>&1 | grep -v WARNING | sed -n '/Phase 3/,$p')"
      rid="$(echo "$out" | grep -m1 "id:" | sed "s/.*id: '\([0-9]*\)'.*/\1/")"
      lvl="$(echo "$out" | grep -m1 "level:" | sed "s/.*level: '\([0-9]*\)'.*/\1/")"
      desc="$(echo "$out" | grep -m1 "description:" | sed "s/.*description: '\(.*\)'/\1/")"
      printf "%-8s L%-3s %s\n" "${rid:-NONE}" "${lvl:--}" "${desc:-(no rule fired)}"
    done
    ;;
  *)
    echo "usage: $0 setup|up|status|down|decisions|logtest" >&2; exit 2 ;;
esac
