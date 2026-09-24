#!/usr/bin/env bash
# End-to-end check: synthetic SpatialData zarr -> spatialdata_to_punkst.py CLI
# -> transcripts.tsv (-> pts2tiles, if a punkst binary is given via $PUNKST).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export PYTHONWARNINGS=ignore

python3 - "$work/synthetic.zarr" <<EOF
import sys
sys.path.insert(0, "$here")
from make_synthetic_sdata import build
build(sys.argv[1], n_points=500, scale=2.0, seed=11)
EOF

python3 "$here/../spatialdata_to_punkst.py" \
  --sdata "$work/synthetic.zarr" --points-key transcripts \
  --coordinate-system microns --out "$work/transcripts.tsv"

head -1 "$work/transcripts.tsv" | grep -q '^#x	y	feature$' || { echo "FAIL: header"; exit 1; }
n=$(tail -n +2 "$work/transcripts.tsv" | wc -l | tr -d ' ')
[ "$n" -eq 500 ] || { echo "FAIL: expected 500 rows, got $n"; exit 1; }
awk -F'\t' 'NR>1 && NF!=3 {bad=1} END{exit bad}' "$work/transcripts.tsv" || { echo "FAIL: column count"; exit 1; }
# microns = 2 x pixels in [0,100] -> all coordinates within [0,200]
awk -F'\t' 'NR>1 && ($1<0||$1>200||$2<0||$2>200){bad=1} END{exit bad}' "$work/transcripts.tsv" || { echo "FAIL: coordinate range"; exit 1; }
echo "PASS: converter CLI ($n rows)"

if [ -n "${PUNKST:-}" ]; then
  "$PUNKST" pts2tiles --in-tsv "$work/transcripts.tsv" \
    --icol-x 0 --icol-y 1 --icol-feature 2 \
    --tile-size 50 --temp-dir "$work/tmp" --out-prefix "$work/transcripts.tiled"
  test -s "$work/transcripts.tiled.tsv" && test -s "$work/transcripts.tiled.index"
  echo "PASS: pts2tiles"
else
  echo "SKIP: pts2tiles (set PUNKST=/path/to/punkst to enable)"
fi
