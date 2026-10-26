"""D-044 PubChem slot merge (casmi.kaggle.predict.interleave / top_unique)."""
from casmi.kaggle.predict import interleave, top_unique


def test_interleave():
    b, bk, p, pk = ['A', 'B', 'C'], ['ka', 'kb', 'kc'], ['X', 'Y'], ['kx', 'ky']
    assert interleave(b, bk, p, pk, (2, 4)) == ['A', 'X', 'B', 'Y', 'C']
    assert interleave(b, bk, [], [], (2, 4)) == b
    assert interleave(b, bk, ['B', 'Y'], ['kb', 'ky'], (1,)) == ['Y', 'A', 'B', 'C']  # already in base: dropped
    assert interleave([], [], p, pk, (4,)) == p  # base exhausted: pc fills
    assert len(interleave(list('abcdefghijklmnopqrstuvwxyz'), list('abcdefghijklmnopqrstuvwxyz'), p, pk, (4, 8))) == 25


def test_top_unique():
    smi, keys = top_unique([(1.0, 'k1', 'a'), (3.0, 'k2', 'b'), (2.0, 'k2', 'c'), (0.5, 'k3', 'd')], n=2)
    assert smi == ['b', 'a'] and keys == ['k2', 'k1']
