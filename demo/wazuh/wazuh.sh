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
FS_WAZUH="$REPO/finsentinel/deploy/wazuh"
ENV_DIR="$HOME/finsentinel-wazuh"
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

point_logs_at_demo() {
  mkdir -p "$ENV_DIR" "$SIEM_DIR" "$DECISIONS_DIR"
  touch "$SIEM_DIR/finsentinel-siem.jsonl" "$SIEM_DIR/fl-coordinator.jsonl"
  if [ ! -f "$ENV_FILE" ]; then
    cp "$FS_WAZUH/.env.example" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
  fi
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
  cat "$src_conf" "$DEMO_WAZUH/manager_overlay.conf" > "$LINUX_DIR/ossec.conf"
  cp "$DEMO_WAZUH/fl_decisions.xml" "$LINUX_DIR/fl_decisions.xml"
  tr -d '\r' < "$DEMO_WAZUH/fl-decision" > "$LINUX_DIR/fl-decision"
  chmod 0644 "$LINUX_DIR/ossec.conf" "$LINUX_DIR/fl_decisions.xml"
  chmod 0755 "$LINUX_DIR/fl-decision"
}

api_ok() {
  local user pass
  user="$(grep -E '^API_USERNAME=' "$ENV_FILE" | cut -d= -f2-)"; user="${user:-wazuh-wui}"
  pass="$(grep -E '^API_PASSWORD=' "$ENV_FILE" | cut -d= -f2-)"
  curl -s -k -m 5 -X POST -u "$user:$pass" https://127.0.0.1:55000/security/user/authenticate | grep -q '"token"'
}

case "${1:-status}" in
  setup)
    point_logs_at_demo
    echo "[fl-demo] Running FinSentinel's setup-wsl.sh (Docker Engine, certs, stack) ..."
    bash "$FS_WAZUH/setup-wsl.sh"
    stage_demo_files
    compose up -d
    echo "[fl-demo] Setup done. The dashboard needs a few more minutes: https://localhost"
    ;;
  up)
    point_logs_at_demo
    stage_demo_files
    if ! command -v docker >/dev/null 2>&1; then echo "Docker not installed — run: $0 setup" >&2; exit 2; fi
    compose up -d
    docker_cmd exec wazuh-manager bash -c \
      'mkdir -p /var/ossec/etc/shared/default && touch /var/ossec/etc/shared/ar.conf && chown -R root:wazuh /var/ossec/etc/shared && chown root:wazuh /var/ossec/active-response/bin/fl-decision 2>/dev/null; true' || true
    echo -n "[fl-demo] waiting for the Wazuh manager API"
    for _ in $(seq 1 120); do
      if api_ok; then echo " — up"; exit 0; fi
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
    # Feeds sample events through the live ruleset and shows which rule fires.
    for ev in \
      '{"source":"finsentinel","event_type":"security.anomaly_clean","severity":"info","client_id":"phone-01","update_id":"phone-01-r0001-aaaa"}' \
      '{"source":"finsentinel","event_type":"security.anomaly_detected","severity":"high","client_id":"phone-03","update_id":"phone-03-r0002-bbbb"}' \
      '{"source":"finsentinel","event_type":"security.anomaly_detected","severity":"critical","client_id":"phone-03","update_id":"phone-03-r0003-cccc"}' \
      '{"source":"finsentinel","event_type":"security.replay_detected","severity":"high","client_id":"phone-04","update_id":"phone-04-r0003-dddd"}' \
      '{"source":"finsentinel","event_type":"security.signature_verification_failed","severity":"high","client_id":"phone-02","update_id":"x"}' \
      '{"source":"finsentinel","event_type":"security.payload_integrity_failed","severity":"high","client_id":"phone-04","update_id":"y"}' \
      '{"source":"fl-coordinator","event_type":"fl.drift_dropout","severity":"low","client_id":"phone-05"}'; do
      echo "== $ev"
      echo "$ev" | docker_cmd exec -i wazuh-manager /var/ossec/bin/wazuh-logtest -q 2>&1 | grep -E "id:|level:|description:|firedtimes|Alert to be" || true
    done
    ;;
  *)
    echo "usage: $0 setup|up|status|down|decisions|logtest" >&2; exit 2 ;;
esac
