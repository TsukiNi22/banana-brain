#!/usr/bin/env bash
# Build the library and the lab server, then serve the lab page: ./lab/server.sh [port] (default 8765)
# The page opened on http://localhost:<port>/ runs its networks with the library (on the gpu when there is one):
# choose "Server" in its Engine menu (the page falls back on its JavaScript engine when the server is down)
set -euo pipefail

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$LAB/.." && pwd)"
PORT="${1:-8765}"
BUILD="$LAB/server/build"

# the library (static, optimized), then the server linked on it
cmake -S "$ROOT" -B "$BUILD/lib" -DCMAKE_BUILD_TYPE=Optimized -DBUILD_TESTS=OFF > /dev/null
cmake --build "$BUILD/lib" --parallel "$(nproc)" > /dev/null
cmake -S "$LAB/server" -B "$BUILD/server" -DBRAIN_BUILD="$BUILD/lib" > /dev/null
cmake --build "$BUILD/server" --parallel "$(nproc)" > /dev/null

# the page is built from lab/src first (the server serves the file of the disk)
"$LAB/build.sh" > /dev/null
exec "$LAB/server/lab_server" --port "$PORT" --page "$LAB/network-lab.html"
