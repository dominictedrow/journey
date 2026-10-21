"""Packed Morgan fingerprints shared by shard targets (P2.5), the candidate pool (P4.2) and inference."""
import numpy as np

FP_BITS = 2048


def fingerprints(smiles):
    """SMILES list -> uint8[n, 256] packed Morgan r2/2048; all-zero row if RDKit can't parse."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    out = np.zeros((len(smiles), FP_BITS // 8), np.uint8)
    for j, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[j] = np.packbits(gen.GetFingerprintAsNumPy(mol))
    return out
