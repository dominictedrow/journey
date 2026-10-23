#!/usr/bin/env bash
# P4.1 (D-024): raw candidate-pool sources -> data/external/{coconut,lotus,pubchem}/. Resumable (curl -C -).
set -euo pipefail
cd ~/envenda
get() { mkdir -p "$(dirname "$2")"; echo "$(date +%T) $2"; curl -fsSL --retry 5 -C - -o "$2" "$1"; ls -l "$2"; }
get https://coconut.s3.uni-jena.de/prod/downloads/2026-10/coconut_csv-10-2026.zip data/external/coconut/coconut_csv-10-2026.zip
get "https://zenodo.org/records/19360665/files/260413_frozen_metadata.csv.gz?download=1" data/external/lotus/260413_frozen_metadata.csv.gz
for f in CID-SMILES.gz CID-Mass.gz; do
  get https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/$f data/external/pubchem/$f
done
echo "$(date +%T) done"
