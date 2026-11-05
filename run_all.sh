#!/usr/bin/env bash
# Full verification run: pytest + control scenarios + tables. Exit code 0 = every row of table 3 is pass.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
rm -rf results logs
mkdir -p results logs
"$PY" tools/make_table3.py 2>&1 | tee results/run_all.txt
code=${PIPESTATUS[0]}
echo "exit code: $code" | tee -a results/run_all.txt
exit $code
