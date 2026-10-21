import numpy as np

from casmi.data.loader import SPEC_KEYS, ShardDataset

SHAPES = {'mz': (np.float32, (128,)), 'inten': (np.float16, (128,)), 'peakfeat': (np.float16, (128, 15)),
          'npeaks': (np.uint16, ()), 'prec': (np.float64, ()), 'neutral_mass': (np.float64, ()), 'ce': (np.float32, (2,)),
          'iso': (np.float32, (6,)), 'adduct': (np.uint16, ()), 'instrument': (np.uint8, ()), 'polarity': (np.uint8, ()),
          'lib': (np.uint8, ()), 'formulas': (np.int16, (16, 10)), 'nformulas': (np.uint8, ()),
          'formula_score': (np.float32, (16,)), 'formula_ppm': (np.float32, (16,)),
          'formula_iso_score': (np.float32, (16,)), 'formula_rdbe': (np.float32, (16,))}


def fake_shard(tmp_path, n_mol=50, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 30, n_mol)
    off = np.append(0, np.cumsum(counts))
    n = off[-1]
    d = tmp_path / 'train'
    d.mkdir()
    for k, (dt, shp) in SHAPES.items():
        np.save(d / f'{k}.npy', np.zeros((n,) + shp, dt))
    mol_of = np.repeat(np.arange(n_mol), counts)
    np.save(d / 'prec.npy', mol_of.astype(np.float64))  # prec encodes the owning molecule
    np.save(d / 'npeaks.npy', rng.integers(1, 129, n).astype(np.uint16))
    np.save(d / 'mol_offsets.npy', off)
    np.save(d / 'mol_fp.npy', np.repeat(np.arange(n_mol, dtype=np.uint8)[:, None], 256, 1))
    np.save(d / 'mol_formula.npy', np.zeros((n_mol, 10), np.int16))
    return counts


def test_epoch_covers_each_molecule_once(tmp_path):
    counts = fake_shard(tmp_path)
    ds = ShardDataset(str(tmp_path), 'train', spec_per_batch=40, max_spec_per_mol=16, block_mols=7, seed=1)
    for _ in range(2):  # two epochs, different shuffles
        seen = []
        for b in ds:
            assert set(SPEC_KEYS) <= set(b)
            m = b['mol_idx'].numpy()
            seen += m.tolist()
            assert (b['mol_fp'][:, 0].numpy() == m).all()
            assert (b['prec'].numpy() == m[b['seg'].numpy()]).all()  # every spectrum sits with its molecule
            L = b['mz'].shape[1]
            assert L % 8 == 0 and L >= b['npeaks'].max() and b['peakfeat'].shape[1] == L
            assert len(b['npeaks']) <= 40 or len(m) == 1
        assert sorted(seen) == list(range(len(counts)))
        per = np.bincount(np.array(seen), minlength=len(counts))
        assert (per == 1).all()


def test_no_cap_keeps_all_spectra_in_order(tmp_path):
    counts = fake_shard(tmp_path)
    ds = ShardDataset(str(tmp_path), 'train', spec_per_batch=64, max_spec_per_mol=None, shuffle=False)
    assert sum(len(b['npeaks']) for b in ds) == counts.sum()


def test_lib_filter_keeps_mol_alignment(tmp_path):
    fake_shard(tmp_path)
    d = tmp_path / "train"
    n = len(np.load(d / "prec.npy"))
    np.save(d / "lib.npy", (np.arange(n) % 3).astype(np.uint8))
    seen = 0
    for b in ShardDataset(str(tmp_path), "train", spec_per_batch=64, block_mols=16, libs={2}):
        assert (b["lib"] == 2).all()
        mol = b["prec"].long()  # owning molecule of each spectrum
        assert (b["mol_idx"][b["seg"]] == mol).all() and (b["mol_fp"][b["seg"], 0].long() == mol).all()
        seen += len(b["lib"])
    assert seen == int((np.arange(n) % 3 == 2).sum())
