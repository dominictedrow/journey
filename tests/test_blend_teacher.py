import numpy as np
import pandas as pd

from knee.paths import LABELS
from knee.train.train_sup import blend_teacher


def test_blend_teacher():
    idx = pd.Index(["a", "b"], name="StudyInstanceUID")
    lab = pd.DataFrame({k: [1.0, 0.0] for k in LABELS}, index=idx)
    lab.loc["a", "ACL"] = np.nan
    for k in LABELS:
        lab[f"w_{k}"] = [0.2, 1.0]
    t = pd.DataFrame({k: [0.0] for k in LABELS}, index=pd.Index(["a"], name="StudyInstanceUID"))
    out = blend_teacher(lab, t, 0.5)
    assert out.loc["a", "MCL"] == 0.5 and out.loc["a", "ACL"] == 0.0  # blended; NaN report takes the teacher
    assert out.loc["a", "w_MCL"] == 0.6                                # 0.5 * 0.2 + 0.5
    assert out.loc["b"].equals(lab.loc["b"])                          # no teacher row → unchanged
    assert np.isnan(lab.loc["a", "ACL"])                              # input not mutated
