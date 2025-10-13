#!/usr/bin/env bash
set -euo pipefail

IMAGE="$1"

cid=""
cleanup() { if [ -n "${cid}" ]; then docker rm -f "${cid}" >/dev/null 2>&1 || true; fi; }
trap cleanup EXIT

cid=$(docker run -d -p 8000:8000 "${IMAGE}")
sleep 8

curl -f http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}' >/dev/null

docker stop "${cid}" >/dev/null
