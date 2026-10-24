# D-036: download PubChem CID-SID / CID-PMID (resumable), then build pool_v1/popularity.npy
set -e
mkdir -p data/external/pubchem
for f in CID-SID.gz CID-PMID.gz; do
  curl -sS -C - -o data/external/pubchem/$f https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/$f
done
~/miniforge3/envs/casmi/bin/python scripts/pubchem_popularity.py
