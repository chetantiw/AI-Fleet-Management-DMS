"""Device health for the fleet platform (Linux /proc and /sys only, no extra packages)."""
import os
import shutil
import time

_prev = None


def cpu_percent():
    global _prev
    with open("/proc/stat") as fh:
        v = [int(x) for x in fh.readline().split()[1:8]]
    idle, total = v[3] + v[4], sum(v)
    if _prev is None:
        _prev = (idle, total)
        return None
    di, dt = idle - _prev[0], total - _prev[1]
    _prev = (idle, total)
    return round(100.0 * (1 - di / dt), 1) if dt else None


def snapshot(disk_path="/"):
    h = {}
    c = cpu_percent()
    if c is not None:
        h["cpu"] = c
    try:
        mem = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0])
        h["mem"] = round(100.0 * (1 - mem["MemAvailable"] / mem["MemTotal"]), 1)
    except (OSError, KeyError, ValueError):
        pass
    du = shutil.disk_usage(disk_path)
    h["disk"] = round(100.0 * du.used / du.total, 1)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            h["cpu_temp"] = round(int(fh.read()) / 1000.0, 1)
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/uptime") as fh:
            h["uptime_h"] = round(float(fh.read().split()[0]) / 3600, 2)
    except (OSError, ValueError):
        pass
    h["load"] = round(os.getloadavg()[0], 2)
    return h


if __name__ == "__main__":
    snapshot()
    time.sleep(1)
    print(snapshot())
