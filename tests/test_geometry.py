"""Geometry rules shared by the audit, the cache builder and inference."""
import numpy as np

from knee.data import geometry as G

SAG = np.array([0, 1, 0, 0, 0, -1], dtype=float)   # rows run posterior, columns run inferior → normal along x
COR = np.array([1, 0, 0, 0, 0, -1], dtype=float)   # normal along y
AX = np.array([1, 0, 0, 0, 1, 0], dtype=float)     # normal along z


def test_as_vec_parses_dicom_strings_and_rejects_bad_input():
    assert np.allclose(G.as_vec("1\\2\\3", 3), [1, 2, 3])
    assert np.allclose(G.as_vec([0.5, 0.25], 2), [0.5, 0.25])
    assert G.as_vec("1\\2", 3) is None
    assert G.as_vec("a\\b\\c", 3) is None
    assert G.as_vec(None, 3) is None


def test_plane_from_iop():
    assert G.plane_from_iop(SAG) == "Sagittal"
    assert G.plane_from_iop(COR) == "Coronal"
    assert G.plane_from_iop(AX) == "Axial"


def test_order_slices_ignores_file_order():
    # Axial stack at z = 0, 3, 6, 9 given in shuffled order.
    z = np.array([6.0, 0.0, 9.0, 3.0])
    ipp = np.stack([np.zeros(4), np.zeros(4), z], axis=1)
    order = G.order_slices(ipp, AX)
    assert np.allclose(z[order], [0, 3, 6, 9])


def test_order_uses_the_normal_for_oblique_stacks():
    # 10° oblique axial: positions must still come out evenly spaced along the normal.
    a = np.radians(10)
    iop = np.array([1, 0, 0, 0, np.cos(a), np.sin(a)])
    n = G.slice_normal(iop)
    ipp = np.stack([k * 4.0 * n + np.array([5.0, -3.0, 2.0]) for k in (3, 0, 2, 1)])
    pos = G.slice_positions(ipp, iop)
    assert np.allclose(np.sort(pos) - np.sort(pos)[0], [0, 4, 8, 12])
    assert abs(G.obliquity_deg(iop) - 10) < 1e-6


def test_axis_letters_follow_lps():
    assert G.axis_letter(np.array([0.99, 0.1, 0])) == "L"
    assert G.axis_letter(np.array([-0.99, 0.1, 0])) == "R"
    assert G.axis_letter(np.array([0, 0, -1.0])) == "I"
    assert G.axis_letter(np.array([0, 1.0, 0])) == "P"


def test_image_center():
    # 4x2 image (rows=2, cols=4), spacing 1 mm, axial, first pixel at the origin.
    c = G.image_center(np.zeros(3), AX, rows=2, cols=4, pixel_spacing=np.array([1.0, 1.0]))
    assert np.allclose(c, [1.5, 0.5, 0])


def test_laterality_from_center_x():
    assert G.laterality_from_center_x(-80.0) == "R"
    assert G.laterality_from_center_x(75.0) == "L"
    assert G.laterality_from_center_x(5.0, dead_zone_mm=10) == "U"
    assert G.laterality_from_center_x(float("nan")) == "U"


def test_spacing_stats_flags_duplicates_and_gaps():
    s = G.spacing_stats(np.array([0, 3, 6, 6, 9, 15.0]))
    assert s["n_duplicate"] == 1
    assert s["median_mm"] == 3
    assert s["max_gap_ratio"] == 2
