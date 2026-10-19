"""MRR@25 mirroring the official Kaggle scorer (metric/casmi-mean-reciprocal-rank, pulled 2026-10-18).

Matching: strip stereo -> RDKit TautomerEnumerator.Canonicalize -> InChIKey[:14].
Stereo is stripped *before* tautomer canonicalization (a stereocentre can pin a double bond
and change the canonical tautomer). Unparseable guesses are non-matches, not errors.
Solution `smiles` may hold several accepted structures joined by ','.
"""
from functools import lru_cache

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
assert rdkit.__version__ == '2026.03.3', f'scorer pins RDKit 2026.03.3, got {rdkit.__version__}'
MAX_RANK = 25
_taut = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=1_000_000)
def ik14(smiles: str) -> str | None:
    """Tautomer-canonical InChIKey14, or None if RDKit can't handle it."""
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToInchiKey(_taut.Canonicalize(mol))[:14] or None
    except (ValueError, RuntimeError, TypeError):
        return None


def reciprocal_rank(guesses: list[str], accepted: set[str], max_rank: int = MAX_RANK) -> float:
    """1/rank of the first guess whose ik14 is in `accepted` (a set of ik14s), else 0."""
    if len(guesses) > max_rank:
        raise ValueError(f'{len(guesses)} guesses > max_rank {max_rank}')
    for rank, g in enumerate(guesses, 1):
        if ik14(g) in accepted:
            return 1.0 / rank
    return 0.0


def score(solution: dict[str, str], submission: dict[str, str], max_rank: int = MAX_RANK) -> float:
    """solution/submission: molecule_id -> ';'-joined guesses / ','-joined accepted SMILES.
    Missing submission ids score 0, as in the official scorer."""
    rr = []
    for mid, answer in solution.items():
        accepted = {k for k in map(ik14, answer.split(',')) if k}
        if not accepted:
            raise ValueError(f'unparseable solution for {mid}: {answer!r}')
        guesses = [g.strip() for g in submission.get(mid, '').split(';') if g.strip()]
        rr.append(reciprocal_rank(guesses, accepted, max_rank))
    return float(np.mean(rr))
