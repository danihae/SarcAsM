# -*- coding: utf-8 -*-
"""Plot functions on small inputs the GUI produces (short fibres, few groups)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sarcasm.motion import Motion  # noqa: E402
from sarcasm.plotting.plots import Plots  # noqa: E402


def test_fibre_delta_slen_with_fewer_sarcomeres_than_rows(monkeypatch):
    """'Show fibre detail' on a 2-sarcomere fibre failed with 'index 2 is out of bounds'."""
    m = Motion.__new__(Motion)
    m.loi_data = {'delta_slen': np.zeros((2, 10)), 'time': np.arange(10) * 0.1}
    monkeypatch.setattr(Plots, '_shade_contr_loi', staticmethod(lambda ax, motion: None))
    fig, ax = plt.subplots()
    Plots._plot_delta_slen_loi(ax, m, n_rows=6, n_start=1)
    assert len(ax.child_axes) == 1                      # sarcomere 1, the only one after n_start
    plt.close(fig)
