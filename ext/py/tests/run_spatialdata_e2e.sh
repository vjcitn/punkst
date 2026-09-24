#!/usr/bin/env bash
# End-to-end check: synthetic SpatialData zarr -> spatialdata_to_punkst.py CLI
# -> transcripts.tsv, then (if a punkst binary is given via $PUNKST)
# pts2tiles -> tiles2hex -> topic-model.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export PYTHONWARNINGS=ignore

# ---- 1. converter CLI on a small fixture with two coordinate systems ----
python3 - "$work/synthetic.zarr" "$here" <<'EOF'
import sys
sys.path.insert(0, sys.argv[2])
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

if [ -z "${PUNKST:-}" ]; then
  echo "SKIP: pts2tiles/tiles2hex/topic-model (set PUNKST=/path/to/punkst to enable)"
  exit 0
fi

# ---- 2. structured fixture: left half enriched for A* genes, right for B* ----
python3 - "$work/structured.zarr" "$here" <<'EOF'
import sys
sys.path.insert(0, sys.argv[2])
from make_synthetic_sdata import build_structured
build_structured(sys.argv[1], seed=5)
EOF
python3 "$here/../spatialdata_to_punkst.py" \
  --sdata "$work/structured.zarr" --points-key transcripts --out "$work/st.tsv"

"$PUNKST" pts2tiles --in-tsv "$work/st.tsv" \
  --icol-x 0 --icol-y 1 --icol-feature 2 \
  --tile-size 100 --temp-dir "$work/tmp1" --out-prefix "$work/st.tiled"
test -s "$work/st.tiled.tsv" && test -s "$work/st.tiled.index"
echo "PASS: pts2tiles"

"$PUNKST" tiles2hex --in-tsv "$work/st.tiled.tsv" --in-index "$work/st.tiled.index" \
  --feature-dict "$work/st.tiled.features.tsv" \
  --icol-x 0 --icol-y 1 --icol-feature 2 \
  --hex-grid-dist 12 --min-count 5 \
  --out "$work/hex.txt" --randomize --seed 1 --temp-dir "$work/tmp2"
test -s "$work/hex.txt" && test -s "$work/hex.json"
echo "PASS: tiles2hex"

"$PUNKST" topic-model --in-data "$work/hex.txt" --in-meta "$work/hex.json" \
  --features "$work/st.tiled.features.tsv" \
  --n-topics 2 --n-epochs 5 --min-count-train 5 \
  --out-prefix "$work/lda" --transform --threads 2 --seed 1
test -s "$work/lda.model.tsv" && test -s "$work/lda.results.tsv"

# The two topics must be dominated by different gene groups (A* vs B*).
python3 - "$work/lda.model.tsv" <<'EOF'
import sys
import pandas as pd
m = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
assert m.shape[1] == 2, m.shape
frac_a = [m.loc[m.index.str.startswith("A"), k].sum() / m[k].sum() for k in m.columns]
assert max(frac_a) > 0.7 and min(frac_a) < 0.3, f"topics did not separate A/B genes: {frac_a}"
print(f"A-gene fraction per topic: {[round(f, 3) for f in frac_a]}")
EOF
echo "PASS: topic-model"
