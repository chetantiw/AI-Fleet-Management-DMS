#!/usr/bin/env python3
"""
One-time setup of ThingsBoard CE for the fleet DMS (safe to re-run; it updates what exists).

  python3 tb_setup.py --url http://localhost:8080 --user tenant@example.com --password ... \
      --vehicle VEH-102 --stream-view http://iot.example.com:8889/veh-102/

Creates / updates, as the tenant administrator:
  1. Rule chain "DMS Events"   - every DMS event with a severity becomes an alarm
                                  (type = event name, details = driver, place, speed, count);
                                  then everything continues to the Root Rule Chain (saved as usual).
  2. Device profile "DMS Vehicle" using that rule chain.
  3. The vehicle device (one per run; run again with another --vehicle for more vehicles),
     its default settings (shared attributes) and the stream viewer URL.
  4. Dashboard "Fleet DMS": fleet map + vehicles table + alarms, and a per-vehicle page with live
     video, start/stop buttons, driver state, events with snapshots, route and speed/tiredness charts.

Prints the device access token to put in the vehicle's device.json.
Widget defaults are read from your ThingsBoard server, so it works on 4.0 and later.
"""
import argparse
import copy
import json
import random
import sys
import uuid

import requests

PROFILE = "DMS Vehicle"
# alias ids must look like UUIDs, or the map widgets silently show nothing
ALIAS_FLEET = "5d0c1f3a-7b2e-4c11-9a6e-0f1e2d3c4b5a"
ALIAS_VEHICLE = "8e4b2a6c-1d3f-4e57-b8a9-6c5d4e3f2a1b"
CHAIN = "DMS Events"
DASHBOARD = "Fleet DMS"
SHARED_DEFAULTS = {"telemetry_interval_s": 5, "stream_max_minutes": 10, "stream_bitrate_k": 400,
                   "stream_width": 640, "event_snapshot": True, "overspeed_kmh": 60}

ALARM_FILTER = ("return msgType == 'POST_TELEMETRY_REQUEST' && msg.event != null && msg.severity != null;")
ALARM_DETAILS = """var details = {};
if (metadata.prevAlarmDetails != null) {
    details = JSON.parse(metadata.prevAlarmDetails);
    metadata.remove('prevAlarmDetails');
}
var n = 1;
if (details.count != null) {
    n = details.count + 1;
}
return {
    "count": n,
    "driver": msg.event_driver,
    "details": msg.event_details,
    "latitude": msg.latitude,
    "longitude": msg.longitude,
    "speed": msg.speed
};"""


class TB:
    def __init__(self, url, user, password):
        self.url = url.rstrip("/")
        r = requests.post(self.url + "/api/auth/login", json={"username": user, "password": password}, timeout=20)
        r.raise_for_status()
        self.h = {"X-Authorization": "Bearer " + r.json()["token"]}

    def get(self, path, **params):
        r = requests.get(self.url + path, headers=self.h, params=params, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json() if r.content else None

    def post(self, path, body, **params):
        r = requests.post(self.url + path, headers=self.h, json=body, params=params, timeout=30)
        if not r.ok:
            sys.exit(f"POST {path} failed: {r.status_code} {r.text[:500]}")
        return r.json() if r.content else None

    def find(self, path, name, **params):
        page = self.get(path, pageSize=100, page=0, textSearch=name, **params) or {"data": []}
        return next((x for x in page["data"] if x["name"] == name), None)


# ------------------------------------------------------------------ rule chain + profile
def setup_rule_chain(tb):
    root = tb.find("/api/ruleChains", "Root Rule Chain", type="CORE")
    if root is None:
        sys.exit("Root Rule Chain not found")
    chain = tb.find("/api/ruleChains", CHAIN, type="CORE") or tb.post("/api/ruleChain", {
        "name": CHAIN, "type": "CORE", "debugMode": False,
        "additionalInfo": {"description": "Fleet DMS: turns DMS events into alarms, then forwards to Root"}})
    meta = {
        "ruleChainId": chain["id"], "firstNodeIndex": 0,
        "nodes": [
            {"type": "org.thingsboard.rule.engine.filter.TbJsFilterNode", "name": "Is DMS alarm event",
             "configuration": {"scriptLang": "TBEL", "tbelScript": ALARM_FILTER,
                               "jsScript": ALARM_FILTER.replace("!= null", "!== undefined")},
             "additionalInfo": {"layoutX": 300, "layoutY": 150}},
            {"type": "org.thingsboard.rule.engine.action.TbCreateAlarmNode", "name": "Create DMS alarm",
             "configuration": {"alarmType": "$[event]", "scriptLang": "TBEL", "alarmDetailsBuildTbel": ALARM_DETAILS,
                               "alarmDetailsBuildJs": "return {};", "severity": "$[severity]",
                               "dynamicSeverity": True, "propagate": False, "propagateToOwner": False,
                               "propagateToTenant": False, "useMessageAlarmData": False,
                               "overwriteAlarmDetails": False, "relationTypes": []},
             "additionalInfo": {"layoutX": 600, "layoutY": 80}},
            {"type": "org.thingsboard.rule.engine.flow.TbRuleChainInputNode", "name": "To Root Rule Chain",
             "configuration": {"ruleChainId": root["id"]["id"], "forwardMsgToDefaultRuleChain": False},
             "additionalInfo": {"layoutX": 600, "layoutY": 250}},
        ],
        "connections": [
            {"fromIndex": 0, "toIndex": 1, "type": "True"},
            {"fromIndex": 0, "toIndex": 2, "type": "True"},
            {"fromIndex": 0, "toIndex": 2, "type": "False"},
            {"fromIndex": 0, "toIndex": 2, "type": "Failure"},
        ],
        "ruleChainConnections": None,
    }
    old = tb.get(f"/api/ruleChain/{chain['id']['id']}/metadata")
    if old and old.get("version") is not None:
        meta["version"] = old["version"]
    tb.post("/api/ruleChain/metadata", meta)
    return chain


def setup_profile(tb, chain):
    infos = tb.get("/api/deviceProfileInfos", pageSize=100, page=0, textSearch=PROFILE) or {"data": []}
    info = next((x for x in infos["data"] if x["name"] == PROFILE), None)
    if info:
        prof = tb.get(f"/api/deviceProfile/{info['id']['id']}")
    else:
        prof = {"name": PROFILE, "type": "DEFAULT", "transportType": "DEFAULT", "provisionType": "DISABLED",
                "description": "In-vehicle edge device: GPS + DMS + face recognition + alcohol sensor",
                "profileData": {"configuration": {"type": "DEFAULT"},
                                "transportConfiguration": {"type": "DEFAULT"},
                                "provisionConfiguration": {"type": "DISABLED", "provisionDeviceSecret": None},
                                "alarms": None}}
    prof["defaultRuleChainId"] = chain["id"]
    # Default (Main) queue. "SequentialByOriginator" would stop the rare duplicate alarm when a burst
    # of same-type events arrives after a network outage, but it processed only ~1 message/s in tests,
    # far too slow to catch up after an outage.
    prof["defaultQueueName"] = None
    return tb.post("/api/deviceProfile", prof)


def setup_device(tb, profile, name, view_url, hls_url):
    dev = tb.get("/api/tenant/devices", deviceName=name)
    if dev is None:
        dev = tb.post("/api/device", {"name": name, "label": name, "deviceProfileId": profile["id"]})
    elif dev["deviceProfileId"]["id"] != profile["id"]["id"]:
        dev["deviceProfileId"] = profile["id"]
        dev = tb.post("/api/device", dev)
    did = dev["id"]["id"]
    shared = dict(SHARED_DEFAULTS)
    have = {a["key"] for a in tb.get(f"/api/plugins/telemetry/DEVICE/{did}/values/attributes/SHARED_SCOPE") or []}
    shared = {k: v for k, v in shared.items() if k not in have}    # never overwrite settings changed later
    if shared:
        tb.post(f"/api/plugins/telemetry/DEVICE/{did}/attributes/SHARED_SCOPE", shared)
    srv = {"inactivityTimeout": 120000}
    if view_url:
        srv["stream_view_url"] = view_url
    if hls_url:
        srv["stream_hls_url"] = hls_url
    tb.post(f"/api/plugins/telemetry/DEVICE/{did}/attributes/SERVER_SCOPE", srv)
    cred = tb.get(f"/api/device/{did}/credentials")
    return dev, cred["credentialsId"]


# ------------------------------------------------------------------ dashboard
class Dash:
    """Builds a dashboard from the widget defaults of the target server."""

    def __init__(self, tb):
        self.tb, self.cache, self.widgets = tb, {}, {}
        self.layouts = {}

    def wtype(self, fqn):
        if fqn not in self.cache:
            w = self.tb.get("/api/widgetType", fqn="system." + fqn)
            if w is None:
                sys.exit(f"widget system.{fqn} not found on this ThingsBoard version")
            self.cache[fqn] = w
        return self.cache[fqn]

    def add(self, state, fqn, title, col, row, sx, sy, patch):
        w = self.wtype(fqn)
        cfg = json.loads(w["descriptor"]["defaultConfig"])
        cfg["title"] = title
        cfg["showTitle"] = bool(title)
        patch(cfg)
        wid = str(uuid.uuid4())
        self.widgets[wid] = {"id": wid, "typeFullFqn": "system." + fqn, "type": w["descriptor"]["type"],
                             "sizeX": sx, "sizeY": sy, "row": 0, "col": 0, "config": cfg}
        self.layouts.setdefault(state, {})[wid] = {"sizeX": sx, "sizeY": sy, "row": row, "col": col}
        return wid


def key(name, label=None, kind="timeseries", **settings):
    return {"name": name, "type": kind, "label": label or name, "color": "#2196f3", "settings": settings,
            "_hash": random.random()}


def entity_ds(alias, keys, latest=None):
    d = {"type": "entity", "name": None, "entityAliasId": alias, "filterId": None, "dataKeys": keys}
    if latest is not None:
        d["latestDataKeys"] = latest
    return d


STATUS_MD = """<div class="dms">
<div class="row"><span>Driver</span><b>${driver}</b></div>
<div class="row"><span>Identity</span><b>${driver_status}</b></div>
<div class="row"><span>Seat belt</span><b>${seatbelt}</b></div>
<div class="row"><span>Eyes</span><b>${eyes}</b></div>
<div class="row"><span>Head</span><b>${head}</b></div>
<div class="row"><span>Phone</span><b>${phone}</b></div>
<div class="row"><span>Tiredness (PERCLOS)</span><b>${perclos:0} %</b></div>
<div class="row"><span>Speed</span><b>${speed:0} km/h</b></div>
<div class="row"><span>Last event</span><b>${event}</b></div>
</div>"""
STATUS_CSS = (".dms{padding:12px 16px;font-size:15px}.dms .row{display:flex;justify-content:space-between;"
              "border-bottom:1px solid rgba(0,0,0,.08);padding:6px 0}.dms span{color:rgba(0,0,0,.54)}")
HEALTH_MD = """<div class="dms">
<div class="row"><span>CPU</span><b>${cpu:0} %</b></div>
<div class="row"><span>CPU temperature</span><b>${cpu_temp:0} °C</b></div>
<div class="row"><span>Memory</span><b>${mem:0} %</b></div>
<div class="row"><span>Storage</span><b>${disk:0} %</b></div>
<div class="row"><span>Messages waiting (no network)</span><b>${buffered:0}</b></div>
<div class="row"><span>Software</span><b>${fw_version}</b></div>
</div>"""
VIDEO_FN = """var d = data && data[0] ? data[0] : {};
var url = d.stream_view_url;
var on = d.stream_active === true || d.stream_active === 'true';
if (!url) {
  return '<div class="novid"><p>No media server URL set for this vehicle (server attribute stream_view_url).</p></div>';
}
if (!on) {
  return '<div class="novid"><p>Live video is off.<br>Press <b>Start live video</b> below: it starts in a few seconds and stops by itself after the set time.</p></div>';
}
// The media server lets in only people logged in to ThingsBoard: pass this user's login token.
var token = '';
try { token = localStorage.getItem('jwt_token') || ''; } catch (e) {}
var src = url + (url.indexOf('?') >= 0 ? '&' : '?') + 'token=' + encodeURIComponent(token);
return '<iframe src="' + src + '" allow="autoplay; fullscreen"></iframe>';"""
VIDEO_CSS = ("iframe{width:100%;height:100%;border:0;background:#000}.novid{height:100%;display:flex;"
             "align-items:center;justify-content:center;text-align:center;padding:24px;color:rgba(0,0,0,.54)}")
IMG_CELL = ("return value ? '<img src=\"' + value + '\" style=\"height:72px;border-radius:4px\">' : '';")


def build_dashboard(tb, device):
    d = Dash(tb)
    A_FLEET, A_VEH = ALIAS_FLEET, ALIAS_VEHICLE
    aliases = {
        A_FLEET: {"id": A_FLEET, "alias": "All vehicles", "filter": {
            "type": "deviceType", "resolveMultiple": True, "deviceTypes": [PROFILE], "deviceNameFilter": ""}},
        A_VEH: {"id": A_VEH, "alias": "Vehicle", "filter": {
            "type": "stateEntity", "resolveMultiple": False, "stateEntityParamName": None,
            "defaultStateEntity": {"entityType": "DEVICE", "id": device["id"]["id"]}}},
    }
    open_vehicle = {"name": "Open vehicle", "icon": "more_horiz", "type": "openDashboardState",
                    "targetDashboardStateId": "vehicle", "setEntityId": True, "stateEntityParamName": None,
                    "openInSeparateDialog": False, "openInPopover": False, "id": str(uuid.uuid4())}

    # ----- fleet page -----
    def fleet_map(c):
        m = c["settings"]["markers"][0]
        m.update(dsType="entity", dsEntityAliasId=A_FLEET, dsLabel="Vehicles", additionalDataKeys=[
            key("driver"), key("speed"), key("seatbelt"), key("event")])
        m["xKey"] = key("latitude")
        m["yKey"] = key("longitude")
        m["label"] = {"show": True, "type": "pattern", "pattern": "${entityName}"}
        m["tooltip"].update(type="pattern", pattern=(
            "<b>${entityName}</b><br/>Driver: ${driver}<br/>Speed: ${speed:0} km/h<br/>"
            "Seat belt: ${seatbelt}<br/>Last event: ${event}"))
        m["markerShape"]["color"] = {"type": "constant", "color": "#307FE5"}
        c["settings"]["markers"] = [m]
        c["settings"].pop("polygons", None)
        c["settings"].pop("circles", None)
        c["settings"]["fitMapBounds"] = True
    d.add("default", "map", "Fleet map", 0, 0, 14, 9, fleet_map)

    def vehicles_table(c):
        c["datasources"] = [entity_ds(A_FLEET, [
            key("name", "Vehicle", "entityField"), key("active", "Online", "attribute"),
            key("driver", "Driver"), key("speed", "Speed km/h"), key("seatbelt", "Seat belt"),
            key("perclos", "PERCLOS %"), key("event", "Last event")])]
        c["settings"]["entitiesTitle"] = "Vehicles"
        c["actions"] = {"rowClick": [open_vehicle]}
    d.add("default", "cards.entities_table", "Vehicles", 14, 0, 10, 9, vehicles_table)

    def fleet_alarms(c):
        c["alarmSource"]["entityAliasId"] = A_FLEET
        c["alarmSource"]["type"] = "entity"
        c["alarmSearchStatus"] = "ACTIVE"
        c["settings"]["alarmsTitle"] = "Active DMS alarms"
        c["timewindow"] = {"realtime": {"interval": 1000, "timewindowMs": 7 * 86400000}}
    d.add("default", "alarm_widgets.alarms_table", "Active DMS alarms", 0, 9, 24, 7, fleet_alarms)

    # ----- vehicle page -----
    def video(c):
        c["datasources"] = [entity_ds(A_VEH, [key("stream_view_url", kind="attribute"),
                                              key("stream_active", kind="attribute")])]
        c["settings"].update(useMarkdownTextFunction=True, markdownTextFunction=VIDEO_FN,
                             applyDefaultMarkdownStyle=False, markdownCss=VIDEO_CSS)
        c["padding"] = "0px"
    d.add("vehicle", "cards.markdown_card", "Live video", 0, 0, 8, 11, video)

    def button(label, icon, method, params, color="#3F52DD"):
        def p(c):
            c["datasources"] = [entity_ds(A_VEH, [])]
            c["targetDevice"] = {"type": "entity", "entityAliasId": A_VEH}
            c["settings"] = {
                "appearance": {"type": "outlined", "autoScale": True, "showLabel": True, "label": label,
                               "showIcon": True, "icon": icon, "iconSize": 24, "iconSizeUnit": "px",
                               "mainColor": color, "backgroundColor": "#FFFFFF",
                               "customStyle": {"enabled": None, "hovered": None, "pressed": None,
                                               "activated": None, "disabled": None}},
                "onClickState": {"action": "EXECUTE_RPC",
                                 "executeRpc": {"method": method, "requestTimeout": 10000,
                                                "requestPersistent": False, "persistentPollingInterval": 1000},
                                 "valueToData": {"type": "CONSTANT", "constantValue": params}}}
        return p
    d.add("vehicle", "command_button", "", 0, 11, 4, 1, button("Start live video", "videocam", "stream_start",
                                                               {"minutes": 5}))
    d.add("vehicle", "command_button", "", 4, 11, 4, 1, button("Stop", "videocam_off", "stream_stop", {}, "#D12730"))

    def status(c):
        c["datasources"] = [entity_ds(A_VEH, [key(k) for k in (
            "driver", "driver_status", "seatbelt", "eyes", "head", "phone", "perclos", "speed", "event")])]
        c["settings"].update(useMarkdownTextFunction=False, markdownTextPattern=STATUS_MD,
                             applyDefaultMarkdownStyle=False, markdownCss=STATUS_CSS)
    d.add("vehicle", "cards.markdown_card", "Driver now", 8, 0, 6, 7, status)

    def health(c):
        c["datasources"] = [entity_ds(A_VEH, [key(k) for k in ("cpu", "cpu_temp", "mem", "disk", "buffered")] +
                                      [key("fw_version", kind="attribute")])]
        c["settings"].update(useMarkdownTextFunction=False, markdownTextPattern=HEALTH_MD,
                             applyDefaultMarkdownStyle=False, markdownCss=STATUS_CSS)
    d.add("vehicle", "cards.markdown_card", "Device health", 8, 7, 6, 5, health)

    def route(c):
        t = c["settings"]["trips"][0]
        t.update(dsType="entity", dsEntityAliasId=A_VEH, dsLabel="Route")
        t["xKey"] = key("latitude")
        t["yKey"] = key("longitude")
        c["settings"]["trips"] = [t]
        c["useDashboardTimewindow"] = False
        c["timewindow"] = {"selectedTab": 0, "realtime": {"realtimeType": 0, "interval": 1000, "timewindowMs": 3600000},
                           "aggregation": {"type": "NONE", "limit": 5000}}
    d.add("vehicle", "trip_map", "Route (last hour)", 14, 0, 10, 12, route)

    def events(c):
        img = key("event_image", "Snapshot", useCellContentFunction=True, cellContentFunction=IMG_CELL)
        c["datasources"] = [entity_ds(A_VEH, [key("event", "Event"), key("event_driver", "Driver"),
                                              key("event_details", "Details"), key("severity", "Severity"),
                                              img], latest=[])]
        c["settings"].update(showTimestamp=True, defaultPageSize=10, displayPagination=True,
                             hideEmptyLines=True)
        c["useDashboardTimewindow"] = False
        c["timewindow"] = {"realtime": {"interval": 1000, "timewindowMs": 86400000},
                           "aggregation": {"type": "NONE", "limit": 500}}
    d.add("vehicle", "cards.timeseries_table", "Events (24 h)", 0, 12, 14, 9, events)

    def chart(c):
        sp, pc = copy.deepcopy(c["datasources"][0]["dataKeys"][:2])
        sp.update(name="speed", type="timeseries", label="Speed km/h", color="#307FE5")
        pc.update(name="perclos", type="timeseries", label="Tiredness PERCLOS %", color="#E8752B")
        for k in (sp, pc):
            for f in ("funcBody",):
                k.pop(f, None)
        c["datasources"] = [entity_ds(A_VEH, [sp, pc])]
        c["useDashboardTimewindow"] = False
        c["timewindow"] = {"realtime": {"interval": 1000, "timewindowMs": 3600000},
                           "aggregation": {"type": "NONE", "limit": 5000}}
    d.add("vehicle", "line_chart", "Speed and tiredness (last hour)", 14, 12, 10, 9, chart)

    def veh_alarms(c):
        c["alarmSource"]["entityAliasId"] = A_VEH
        c["alarmSource"]["type"] = "entity"
        c["settings"]["alarmsTitle"] = "Alarms"
        c["timewindow"] = {"realtime": {"interval": 1000, "timewindowMs": 7 * 86400000}}
    d.add("vehicle", "alarm_widgets.alarms_table", "Alarms (7 days)", 0, 21, 24, 7, veh_alarms)

    def layout(state):
        return {"main": {"widgets": d.layouts[state], "gridSettings": {
            "backgroundColor": "#eeeeee", "columns": 24, "margin": 10, "outerMargin": True,
            "backgroundSizeMode": "100%", "autoFillHeight": False, "mobileAutoFillHeight": False,
            "mobileRowHeight": 70, "layoutType": "default"}}}
    return {
        "title": DASHBOARD,
        "configuration": {
            "description": "Fleet DMS: vehicles, driver monitoring alarms and live video",
            "widgets": d.widgets,
            "states": {"default": {"name": "Fleet", "root": True, "layouts": layout("default")},
                       "vehicle": {"name": "${entityName}", "root": False, "layouts": layout("vehicle")}},
            "entityAliases": aliases, "filters": {},
            "timewindow": {"realtime": {"interval": 1000, "timewindowMs": 3600000},
                           "aggregation": {"type": "NONE", "limit": 5000}},
            "settings": {"stateControllerId": "entity", "showTitle": True, "showDashboardsSelect": True,
                         "showEntitiesSelect": True, "showDashboardTimewindow": True,
                         "showDashboardExport": True, "toolbarAlwaysOpen": True},
        },
    }


def setup_dashboard(tb, device):
    body = build_dashboard(tb, device)
    old = tb.find("/api/tenant/dashboards", DASHBOARD)
    if old:
        cur = tb.get(f"/api/dashboard/{old['id']['id']}")
        body["id"], body["version"] = cur["id"], cur.get("version")
        # keep the default vehicle chosen the first time
        a = cur.get("configuration", {}).get("entityAliases", {}).get(ALIAS_VEHICLE)
        if a:
            body["configuration"]["entityAliases"][ALIAS_VEHICLE] = a
    return tb.post("/api/dashboard", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--user", required=True, help="tenant administrator e-mail")
    ap.add_argument("--password", required=True)
    ap.add_argument("--vehicle", required=True, help="device name, e.g. VEH-102")
    ap.add_argument("--stream-view", default="", help="WebRTC player URL of this vehicle on the media server")
    ap.add_argument("--stream-hls", default="", help="HLS player URL (optional)")
    ap.add_argument("--save-dashboard", help="also write the dashboard JSON to this file")
    a = ap.parse_args()
    tb = TB(a.url, a.user, a.password)
    chain = setup_rule_chain(tb)
    print("rule chain:", CHAIN)
    prof = setup_profile(tb, chain)
    print("device profile:", PROFILE)
    dev, token = setup_device(tb, prof, a.vehicle, a.stream_view, a.stream_hls)
    print("device:", a.vehicle, dev["id"]["id"])
    dash = setup_dashboard(tb, dev)
    print("dashboard:", f"{a.url}/dashboards/{dash['id']['id']}")
    if a.save_dashboard:
        with open(a.save_dashboard, "w") as fh:
            json.dump(build_dashboard(tb, dev), fh, indent=1)
    print("\nDevice access token (put it in device.json -> thingsboard.token):\n  " + token)


if __name__ == "__main__":
    main()
