import pytest

from casmi.metric import ik14, reciprocal_rank, score

SOL = {'m_1': 'CCO', 'm_2': 'c1ccccc1'}


# Cases copied from the official scorer's doctests.
def test_rank_positions():
    assert score(SOL, {'m_1': 'CCO;CC(=O)O', 'm_2': 'CC(=O)O;c1ccccc1'}) == 0.75


def test_first_hit_only_and_miss():
    assert score(SOL, {'m_1': 'CC(=O)O', 'm_2': 'c1ccccc1;c1ccccc1;c1ccccc1'}) == 0.5


def test_stereo_free():
    assert score({'m': 'C[C@H](N)C(=O)O'}, {'m': 'CC(N)C(=O)O'}) == 1.0


def test_stereo_pinned_tautomer_regression():
    assert score({'m': 'C=C(C)C1CCC2CCC(=O)C=C2C1'}, {'m': 'C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1'}) == 1.0


def test_too_many_guesses():
    with pytest.raises(ValueError):
        score({'m': 'CCO'}, {'m': 'CCO;CCO;CCO'}, max_rank=2)


def test_bad_solution_raises():
    with pytest.raises(ValueError):
        score({'m': 'not a molecule'}, {'m': 'CCO'})


def test_bad_guess_is_miss_and_missing_id_scores_zero():
    assert reciprocal_rank(['garbage((', 'CCO'], {ik14('CCO')}) == 0.5
    assert score(SOL, {'m_1': 'CCO'}) == 0.5


def test_glucose_stereo_variants_match():
    alpha_d = 'OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O'
    beta_d = 'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O'
    flat = 'OCC1OC(O)C(O)C(O)C1O'
    assert ik14(alpha_d) == ik14(beta_d) == ik14(flat) is not None


def test_keto_enol_tautomers_match():
    assert ik14('CC(=O)C') == ik14('CC(O)=C')


def test_multiple_accepted_answers():
    assert score({'m': 'CCO,c1ccccc1'}, {'m': 'CC(=O)O;c1ccccc1'}) == 0.5
