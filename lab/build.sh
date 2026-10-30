#!/bin/bash
# Build lab/network-lab.html (one self-contained page) from lab/src with the html-style tools of the ai-utils skills
# Usage: lab/build.sh [path of the html-style skill]   (default: ~/.claude/skills/html-style)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STYLE="${1:-$HOME/.claude/skills/html-style}"
WORKER="$(mktemp --suffix=.js)"; trap 'rm -f "$WORKER"' EXIT
python3 - "$HERE/src" "$WORKER" <<'PY'
import json, sys
src, out = sys.argv[1], sys.argv[2]
code = ''.join(open(f'{src}/{name}').read() + '\n' for name in ('model.js', 'lab-tasks.js', 'batch-worker.js'))
open(out, 'w').write('// The code of the batch workers (built by lab/build.sh)\nwindow.LAB_WORKER_SRC = ' + json.dumps(code) + ';\n')
PY
python3 "$STYLE/scripts/new_page.py" "$HERE/src/lab.body.html" -o "$HERE/network-lab.html" --force \
    --css "$HERE/src/viz.css" "$HERE/src/lab.css" --js "$HERE/src/model.js" "$HERE/src/lab-tasks.js" "$HERE/src/batch-worker.js" "$WORKER" "$HERE/src/remote.js" "$HERE/src/lab.js" \
    --set PROJECT=banana-brain TITLE="Network lab" STORAGE_KEY=banana-brain-lab FOOTER="banana-brain network lab (not published)"
python3 "$STYLE/scripts/check_style.py" "$HERE/network-lab.html"
