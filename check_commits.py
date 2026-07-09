import subprocess
import re

out = subprocess.check_output(
    [
        "git",
        "log",
        "--author=VyomKulshrestha",
        "--format=COMMIT %h %cd %s",
        "--date=short",
        "--shortstat",
    ]
).decode("utf-8")
lines = out.split("\n")

commits = []
current_commit = ""
for line in lines:
    if line.startswith("COMMIT "):
        current_commit = line[7:]
    elif "insertions" in line:
        match = re.search(r"(\d+) insertions", line)
        if match:
            commits.append(
                {
                    "commit": current_commit,
                    "ins": int(match.group(1)),
                    "stats": line.strip(),
                }
            )

for c in sorted(commits, key=lambda x: x["ins"], reverse=True)[:10]:
    print(f"{c['commit']} | {c['stats']}")
