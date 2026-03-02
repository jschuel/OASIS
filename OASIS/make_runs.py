rcs = [0.5, 1.0, 2.0]
rns = [0.5, 1.0, 2.0]
ros = [1.0, 2.0, 4.0, 8.0]

def fmt(x):
    s = str(x).replace(".", "p")
    if s.endswith("p0"):
        s = s[:-2]
    return s

lines = []
ids = []
for rc in rcs:
    for rn in rns:
        for ro in ros:
            run_id = f"rc{fmt(rc)}_rn{fmt(rn)}_ro{fmt(ro)}"
            lines.append(f"{rc} {rn} {ro} {run_id}")
            ids.append(run_id)

with open("runs.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
with open("run_ids.txt", "w") as f:
    f.write("\n".join(ids) + "\n")

print(f"Wrote runs.txt ({len(lines)} lines) and run_ids.txt")
