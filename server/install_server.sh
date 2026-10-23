#!/usr/bin/env bash
# Installs the fleet DMS server side next to an existing ThingsBoard CE (run on the ThingsBoard server).
#
#   sudo ./server/install_server.sh iot.mutechautomation.com VEH-102 [VEH-103 ...]
#
# 1. Media server (MediaMTX) + viewer login check, with Docker   -> server/mediamtx
# 2. A random video password per vehicle                          -> server/mediamtx/vehicles.json
# 3. nginx: /live/ on the ThingsBoard HTTPS site                  -> prints what to add
# 4. ThingsBoard: rule chain, device profile, devices, dashboard -> asks for the tenant admin login
# 5. One device.json per vehicle (token + video password filled)  -> server/vehicles/<VEH>.device.json
set -euo pipefail
DOMAIN=${1:?usage: install_server.sh <domain> <VEH-ID> [VEH-ID ...]}
shift
[ $# -ge 1 ] || { echo "give at least one vehicle id, e.g. VEH-102"; exit 1; }
cd "$(dirname "$0")"
HERE=$(pwd)

# ---- 1-2. media server + passwords ----
command -v docker >/dev/null || { echo "Docker is needed: https://docs.docker.com/engine/install/"; exit 1; }
cd mediamtx
[ -f vehicles.json ] || echo '{}' > vehicles.json
python3 - "$@" <<'EOF'
import json, secrets, sys
v = json.load(open("vehicles.json"))
for veh in sys.argv[1:]:
    v.setdefault(veh.lower(), secrets.token_urlsafe(12))
json.dump(v, open("vehicles.json", "w"), indent=1)
EOF
chmod 600 vehicles.json
sed -i "s|^webrtcAdditionalHosts:.*|webrtcAdditionalHosts: [$DOMAIN]|" mediamtx.yml
docker compose up -d
cd "$HERE"

# ---- 3. nginx ----
echo
echo "=== nginx: add this inside the 'server { listen 443 ssl; server_name $DOMAIN; ... }' block,"
echo "=== then: sudo nginx -t && sudo systemctl reload nginx"
sed -n '/^location/,$p' mediamtx/nginx-live.conf
echo
echo "=== firewall: open 1883/tcp (MQTT), 8554/tcp (video from vehicles), 8189/udp and 8189/tcp (WebRTC)"
echo "    e.g.  sudo ufw allow 1883/tcp && sudo ufw allow 8554/tcp && sudo ufw allow 8189"
echo

# ---- 4-5. ThingsBoard ----
python3 -c "import requests" 2>/dev/null || pip3 install requests
read -rp "ThingsBoard tenant admin e-mail: " TBUSER
read -rsp "Password: " TBPASS; echo
TBURL=${TB_URL:-http://127.0.0.1:8080}
mkdir -p vehicles
for VEH in "$@"; do
  path=$(echo "$VEH" | tr 'A-Z' 'a-z')
  out=$(python3 thingsboard/tb_setup.py --url "$TBURL" --user "$TBUSER" --password "$TBPASS" \
        --vehicle "$VEH" --stream-view "https://$DOMAIN/live/$path/")
  echo "$out" | head -4
  token=$(echo "$out" | tail -1 | tr -d ' ')
  pass=$(python3 -c "import json;print(json.load(open('mediamtx/vehicles.json'))['$path'])")
  cat > "vehicles/$VEH.device.json" <<JSON
{
  "vehicle_id": "$VEH",
  "thingsboard": {"host": "$DOMAIN", "port": 1883, "tls": false, "token": "$token"},
  "stream": {
    "path": "$path",
    "publish_url": "rtsp://$path:$pass@$DOMAIN:8554/$path",
    "view_url": "https://$DOMAIN/live/$path/"
  },
  "gnss": {"type": "demo", "speed_kmh": 32},
  "outbox": "tb_outbox.db"
}
JSON
  chmod 600 "vehicles/$VEH.device.json"
  echo "-> server/vehicles/$VEH.device.json  (copy to the vehicle as dms/device.json)"
done
echo
echo "Done. Dashboard: https://$DOMAIN  ->  Dashboards -> Fleet DMS"
