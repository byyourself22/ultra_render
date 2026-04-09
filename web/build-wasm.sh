#!/bin/bash
# Build WASM with correct LIB path for dbghelp.lib
cd "$(dirname "$0")/.."
export LIB="$(pwd);$LIB"

MODE="--dev"
if [ "$1" = "release" ]; then
  MODE=""
fi

wasm-pack build . $MODE --target web --no-default-features --features web
