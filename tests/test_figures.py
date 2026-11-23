"""Tests of the class colors used in figures and maps."""

import subprocess
import sys

import numpy as np

from ml_structural_system.figures import CLASS_COLORS, class_colors


def test_known_classes_keep_their_color():
    colors = class_colors(["M", "CR", "ADO", "S"])
    assert colors == CLASS_COLORS


def test_other_classes_get_tab10_in_alphabetical_order():
    colors = class_colors(["zeta", "alpha", "M", None, np.nan])
    assert set(colors) == {"alpha", "M", "zeta"}
    assert colors["M"] == CLASS_COLORS["M"]
    assert colors["alpha"] == "#1f77b4" and colors["zeta"] == "#ff7f0e"
    assert colors == class_colors(["alpha", "zeta", "M"])  # independent of the order


def test_colors_do_not_change_between_runs():
    code = "from ml_structural_system.figures import class_colors; print(class_colors(['W', 'X', 'ADO']))"
    runs = {
        subprocess.run([sys.executable, "-c", code], env={"PYTHONHASHSEED": seed, "PATH": ""},
                       capture_output=True, text=True, check=True).stdout
        for seed in ("1", "2", "3")
    }
    assert len(runs) == 1
