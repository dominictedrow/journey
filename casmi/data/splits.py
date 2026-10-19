"""P2.3 split roles (D-019). Splits are per tautomer-canonical ik14 (data/processed/v1/splits.parquet)."""

VAL = ('v_lib', 'v_pool', 'v_scaf', 'v_novel')


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
