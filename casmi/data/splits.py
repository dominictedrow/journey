"""P2.3 split roles (D-019). Splits are per tautomer-canonical ik14 (data/processed/v1/splits.parquet)."""

VAL = ('v_lib', 'v_pool', 'v_scaf', 'v_novel')
NP = 'v_np'  # D-035 natural-product fold, carved out of train by scripts/build_np_split.py


def np_keys(root):
    """ik14s of the V-np fold (empty if not built): training and the val library must skip them."""
    import os

    import pyarrow.parquet as pq
    p = f'{root}/{NP}/mols.parquet'
    return pq.read_table(p, columns=['ik14'])['ik14'].to_pylist() if os.path.exists(p) else []


def spectrum_role(split, is_enveda):
    """'train' | 'query' | None (unused). v_lib keeps its non-enveda spectra in train (kind-1 reference)."""
    if split == 'train' or (split == 'v_lib' and not is_enveda):
        return 'train'
    if split in VAL and is_enveda:
        return 'query'
    return None


def in_pool(split):
    """V-novel structures are excluded from every candidate pool."""
    return split != 'v_novel'
