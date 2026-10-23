"""
Vehicle position for the telemetry.

- NmeaGnss: reads a GNSS receiver (u-blox, Quectel LTE modem's GNSS port, ...) that outputs NMEA
  sentences on a serial port; parses RMC (position, speed, heading, fix) and GGA (altitude,
  satellites, HDOP). Needs `pip install pyserial`.
- RouteReplay: drives a fake vehicle along a list of way-points, for bench tests and the recorded
  video demo before the GNSS hardware is fitted.

Both give .fix() -> dict of ThingsBoard telemetry keys, or {"gps_fix": False} with no fix.
"""
import math
import threading
import time

# A loop through Indore (Rajwada - Palasia - Vijay Nagar - LIG - Rajwada), for demos only.
DEMO_ROUTE = [(22.7186, 75.8553), (22.7196, 75.8680), (22.7247, 75.8867), (22.7363, 75.8822),
              (22.7533, 75.8937), (22.7490, 75.9010), (22.7363, 75.8822), (22.7247, 75.8867),
              (22.7196, 75.8680), (22.7186, 75.8553)]


def _dist_m(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def _bearing(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


class RouteReplay:
    def __init__(self, route=DEMO_ROUTE, speed_kmh=32.0):
        self.route, self.v = route, speed_kmh / 3.6
        self.legs = [_dist_m(route[i], route[i + 1]) for i in range(len(route) - 1)]
        self.total = sum(self.legs)
        self.t0 = time.time()

    def fix(self, t=None):
        s = ((t if t is not None else time.time() - self.t0) * self.v) % self.total
        for i, d in enumerate(self.legs):
            if s <= d or i == len(self.legs) - 1:
                a, b, k = self.route[i], self.route[i + 1], min(s / d, 1.0) if d else 0.0
                # ease the speed a little so the speed chart is not a flat line
                spd = self.v * 3.6 * (0.75 + 0.25 * math.sin(math.pi * k))
                return {"gps_fix": True, "latitude": round(a[0] + (b[0] - a[0]) * k, 6),
                        "longitude": round(a[1] + (b[1] - a[1]) * k, 6), "speed": round(spd, 1),
                        "heading": round(_bearing(a, b), 1), "altitude": 553.0, "satellites": 11, "hdop": 0.9}
            s -= d


def _deg(v, hemi):
    if not v:
        return None
    d = int(float(v) / 100)
    x = d + (float(v) - d * 100) / 60
    return -x if hemi in ("S", "W") else x


class NmeaGnss:
    def __init__(self, port="/dev/ttyAMA0", baud=9600):
        import serial  # pyserial
        self.ser = serial.Serial(port, baud, timeout=1)
        self.state = {"gps_fix": False}
        self.last = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                line = self.ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                time.sleep(1)
                continue
            if not line.startswith("$") or "*" not in line:
                continue
            body, cs = line[1:].split("*", 1)
            x = 0
            for ch in body:
                x ^= ord(ch)
            if f"{x:02X}" != cs[:2].upper():
                continue
            f = body.split(",")
            kind = f[0][2:]
            if kind == "RMC" and len(f) >= 9:
                ok = f[2] == "A"
                st = dict(self.state, gps_fix=ok)
                if ok:
                    st.update(latitude=round(_deg(f[3], f[4]), 6), longitude=round(_deg(f[5], f[6]), 6),
                              speed=round(float(f[7] or 0) * 1.852, 1), heading=float(f[8] or 0))
                self.state, self.last = st, time.time()
            elif kind == "GGA" and len(f) >= 10:
                if f[7]:
                    self.state["satellites"] = int(f[7])
                if f[8]:
                    self.state["hdop"] = float(f[8])
                if f[9]:
                    self.state["altitude"] = float(f[9])

    def fix(self, t=None):
        if time.time() - self.last > 5:
            return {"gps_fix": False}
        return dict(self.state)
