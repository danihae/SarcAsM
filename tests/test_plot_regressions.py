# -*- coding: utf-8 -*-
"""Plot functions on small inputs the GUI produces (short groups, few groups)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sarcasm.plotting.plots import Plots  # noqa: E402
from .test_grouped_motion import _fake_structure  # noqa: E402


def test_stacked_delta_slen_with_fewer_sarcomeres_than_rows():
    """The group detail of a 2-sarcomere myofibril failed with 'index 2 is out of bounds'."""
    sarc = _fake_structure(n_tracks=6)
    sarc.data['structure.myofibril.lines'] = [[np.array([0, 1]), np.array([2, 3, 4, 5])]]
    groups = sarc.motion.analyze('myofibril', reference_frame=0)
    g = next(g for g in groups if g.n_members == 2)
    fig, ax = plt.subplots()
    Plots.plot_delta_slen(ax, g, stacked=True, n_rows=6, n_start=1)
    assert [t.get_text() for t in ax.get_yticklabels()] == ['2', 'mean']   # the only sarcomere after n_start
    plt.close(fig)
