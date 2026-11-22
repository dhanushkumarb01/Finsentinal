#!/usr/bin/env bash
# Keep a localhost.run tunnel up (free tier rotates/drops); write the current URL to
# demo/.state/tunnel_url.txt. Usage: bash demo/scripts/tunnel_lhr.sh [local_port]
PORT="${1:-8790}"
STATE="$(cd "$(dirname "$0")/.." && pwd)/.state"
mkdir -p "$STATE"
while true; do
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -R "80:127.0.0.1:${PORT}" nokey@localhost.run 2>&1 |
  while IFS= read -r line; do
    url=$(printf '%s' "$line" | grep -aoE 'https://[a-z0-9.-]+\.lhr\.life' | tail -1)
    if [ -n "$url" ]; then
      echo "$url" > "$STATE/tunnel_url.txt"
      echo "$(date -u +%FT%TZ) TUNNEL $url"
    fi
  done
  echo "$(date -u +%FT%TZ) tunnel dropped; restarting in 3 s"
  sleep 3
done
