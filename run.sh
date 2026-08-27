#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p bin data/input data/output
if [ ! -f bin/edge_detect ]; then
  make
fi
echo '' > run_log.txt
./bin/edge_detect -i data/input -o data/output -n 0 | tee -a run_log.txt
