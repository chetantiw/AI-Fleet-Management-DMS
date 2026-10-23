"""Side-by-side clip: cabin video | 3-D twin.   compare.py page.html clip video.mp4 start end out.mp4"""
import subprocess, sys, os, shutil
from playwright.sync_api import sync_playwright
page, clip, video, t0, t1, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
tmp = out + "_frames"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
NM = "/home/claude/work/viewer/npm/node_modules/three/"
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 760, "height": 760})
    pg.route("https://cdn.jsdelivr.net/npm/three@0.147.0/**", lambda r: r.fulfill(path=NM + r.request.url.split("three@0.147.0/")[1], content_type="application/javascript"))
    pg.route("https://fonts.**", lambda r: r.abort())
    pg.goto("file://" + page); pg.wait_for_timeout(2500)
    pg.evaluate(f"twin.load('{clip}'); twin.setCam('dms'); twin.fast()")
    el = pg.query_selector("#view3d")
    n = int(round((t1 - t0) * 10))
    pg.evaluate(f"twin.renderAt({t0})")
    for i in range(n):
        pg.evaluate(f"twin.stepRender({t0 + i / 10:.2f}, 0.1)")
        el.screenshot(path=f"{tmp}/{i:05d}.png")
    b.close()
# video on the left (same 10 fps), twin on the right, both 960 px tall
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-t", str(t1 - t0), "-i", video,
                "-framerate", "10", "-i", f"{tmp}/%05d.png",
                "-filter_complex", "[0:v]scale=-2:960,fps=10[a];[1:v]scale=-2:960[b];[a][b]hstack=inputs=2,format=yuv420p",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", out], check=True)
shutil.rmtree(tmp)
print(out)
