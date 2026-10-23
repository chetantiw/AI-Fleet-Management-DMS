"""Fill the viewer template with twin streams: build.py out.html name=label=path.json ..."""
import json, sys, math
out, specs = sys.argv[1], sys.argv[2:]
clips = {}
for sp in specs:
    name, label, path = sp.split("=", 2)
    if path == "FAKE":
        fr = []
        for i in range(600):
            t = i / 10
            fr.append(dict(t=t, yaw=round(35 * math.sin(t / 3), 1), pitch=round(10 * math.sin(t / 2), 1), roll=0, eyeL=1 if (i % 40) > 3 else 0,
                           eyeR=1 if (i % 40) > 3 else 0, jaw=round(abs(math.sin(t)), 2), lean=[0, 0], lE=[0.75, -0.95, 0.45], lW=[0.45, -1.15, 1.1],
                           rE=[-0.75, -0.95, 0.45], rW=[-0.45, -1.15, 1.1], armOk=1, face=1, facing=int(abs(35 * math.sin(t / 3)) < 25), closed=0, hidden=0,
                           phone=int(20 < t < 40), mouth=int(45 < t < 50), ear=int(30 < t < 40), belt=1 if t < 25 else 0, perclos=0.05, hand="L" if t < 42 else "R",
                           alert="PHONE CALL" if 31 < t < 34 else "", ev=["PHONE_CALL"] if i == 310 else [], driver="Unknown #1"))
        clips[name] = dict(label=label, meta={}, frames=fr)
    else:
        d = json.load(open(path)); d["label"] = label; clips[name] = d
tpl = open(__file__.replace("build.py", "template.html")).read()
open(out, "w").write(tpl.replace("/*TWIN_DATA*/", json.dumps(dict(clips=clips), separators=(",", ":")).replace("</", "<\\/")))
print(out)
