#!/usr/bin/env bash
set -euo pipefail
TAG="$1"

awk 'found && /^## /{exit} /^##[[:space:]]*\[?'"$TAG"'\]?/{found=1} found' CHANGELOG.md > RELEASE_NOTES.md || true
if [ ! -s RELEASE_NOTES.md ]; then
  cp CHANGELOG.md RELEASE_NOTES.md
fi
