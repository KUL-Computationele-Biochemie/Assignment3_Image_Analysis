"""
conftest.py -- AST-based extraction of student code from marimo notebooks.

marimo notebooks are plain .py files, but student code lives in three
structural positions:

    1. Module-level  (@app.function / @app.class_definition)
       -> ast.walk finds the FunctionDef/ClassDef directly.

    2. Cell-nested   (inside  def _(...):  wrappers created by @app.cell)
       -> ast.walk recurses into every scope and still finds the inner
          FunctionDef/ClassDef.  ast.get_source_segment returns the segment
          starting at the def/class keyword; textwrap.dedent normalises
          any leading whitespace.

    3. Unparsable cells  (app._unparsable_cell(r\"\"\"...\"\"\", name=...))
       -> The code lives inside a raw-string literal argument to a method
          call.  ast.walk never sees a FunctionDef/ClassDef inside it.
          We: (a) find Call nodes whose func is app._unparsable_cell,
          (b) pull the first positional string argument,
          (c) dedent and strip it, (d) try ast.parse.  If it parses we
          exec it.  If not, the student has not yet completed the cell —
          we record None so the fixture can skip cleanly.
"""

from __future__ import annotations

import ast
import textwrap
from enum import Enum
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("Agg")

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDENT_NOTEBOOK = REPO_ROOT / "notebooks" / "lecture_note_ia_student.marimo.py"


def _extract_from_ast(source: str, target_name: str) -> str | None:
    """Walk the full AST and return the dedented source segment of the
    first FunctionDef or ClassDef whose .name == target_name.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == target_name:
                segment = ast.get_source_segment(source, node)
                if segment is not None:
                    return textwrap.dedent(segment)
    return None


def _extract_from_unparsable_cells(source: str) -> dict[str, str | None]:
    """Find every app._unparsable_cell(...) call and attempt to parse its
    string payload.
    """
    tree = ast.parse(source)
    results: dict[str, str | None] = {}

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "_unparsable_cell"):
            continue
        if not call.args or not isinstance(call.args[0], ast.Constant):
            continue
        raw = call.args[0].value
        if not isinstance(raw, str):
            continue

        code = textwrap.dedent(raw).strip()

        try:
            inner_tree = ast.parse(code)
        except SyntaxError:
            fallback = _keyword_name(call)
            if fallback:
                results[fallback] = None
            continue

        for inner_node in ast.iter_child_nodes(inner_tree):
            if isinstance(
                inner_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                results[inner_node.name] = code
                break

    return results


def _keyword_name(call: ast.Call) -> str | None:
    """Extract the symbol name from the name= keyword."""
    for kw in call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
            name = kw.value.value
            if isinstance(name, str):
                return name.lstrip("*")
    return None


def _build_namespace(source: str) -> dict[str, Any]:
    """Extract every student symbol and exec it into a single namespace."""
    from scipy.optimize import curve_fit
    from scipy.integrate import odeint as _odeint
    import numpy as _np
    from pathlib import Path as _Path

    class Nucleotide(Enum):
        ADENINE = 1
        GUANINE = 2
        CYTOSINE = 3
        THYMINE = 4

    import tifffile as _tifffile
    from skimage.feature import blob_dog, blob_log, blob_doh
    from skimage.measure import label
    import math

    ns: dict[str, Any] = {
        "np": np,
        "plt": plt,
        "curve_fit": curve_fit,
        "odeint": _odeint,
        "Nucleotide": Nucleotide,
        "tifffile": _tifffile,
        "blob_dog": blob_dog,
        "blob_log": blob_log,
        "blob_doh": blob_doh,
        "label": label,
        "math": math,
    }

    ast_targets = [
        "find_spots",
        "extract_nucleotides_from_image",
        "full_analysis",
        "segment_cells",
        "calculate_cell_fluorescence",
        "fluorescence_trajectory",
    ]

    for name in ast_targets:
        code = _extract_from_ast(source, name)
        if code is None:
            continue
        try:
            exec(compile(code, f"<student:{name}>", "exec"), ns)
        except Exception:
            pass

    unparsable = _extract_from_unparsable_cells(source)

    return ns


@pytest.fixture(scope="session")
def student_ns() -> dict[str, Any]:
    source = STUDENT_NOTEBOOK.read_text(encoding="utf-8")
    return _build_namespace(source)


def _require(ns: dict[str, Any], name: str):
    if name not in ns:
        pytest.skip(f"'{name}' not yet implemented in student notebook")
    return ns[name]


@pytest.fixture(scope="session")
def find_spots(student_ns):
    return _require(student_ns, "find_spots")


@pytest.fixture(scope="session")
def extract_nucleotides_from_image(student_ns):
    return _require(student_ns, "extract_nucleotides_from_image")


@pytest.fixture(scope="session")
def full_analysis(student_ns):
    return _require(student_ns, "full_analysis")


@pytest.fixture(scope="session")
def segment_cells(student_ns):
    return _require(student_ns, "segment_cells")


@pytest.fixture(scope="session")
def calculate_cell_fluorescence(student_ns):
    return _require(student_ns, "calculate_cell_fluorescence")


@pytest.fixture(scope="session")
def fluorescence_trajectory(student_ns):
    return _require(student_ns, "fluorescence_trajectory")


@pytest.fixture(scope="session")
def four_channel_image():
    """3D image with spots at known channels. Shape: (L, L, 4).
    Channel order: A=0, T=1, C=2, G=3
    """
    L = 256
    image = np.zeros((L, L, 4))

    def makeGaussian(size, fwhm=20, center=None):
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        if center is None:
            x0 = y0 = size // 2
        else:
            x0, y0 = center
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)

    spots = [
        ((50, 50), 0),  # Adenine
        ((100, 80), 1),  # Thymine
        ((150, 120), 2),  # Cytosine
        ((200, 180), 3),  # Guanine
    ]

    for (cx, cy), channel in spots:
        image[:, :, channel] = makeGaussian(L, fwhm=20, center=(cx, cy))

    return image


@pytest.fixture(scope="session")
def four_dim_image():
    """4D image stack with known sequences. Shape: (L, L, 4, n_cycles).
    Channel order: A=0, T=1, C=2, G=3
    """
    L = 256
    n_cycles = 5
    image = np.zeros((L, L, 4, n_cycles))

    def makeGaussian(size, fwhm=20, center=None):
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        if center is None:
            x0 = y0 = size // 2
        else:
            x0, y0 = center
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)

    spots = [
        ((50, 50), [0, 1, 2, 3, 0]),  # Spot 1: A, T, C, G, A
        ((100, 80), [1, 2, 3, 0, 1]),  # Spot 2: T, C, G, A, T
        ((150, 120), [2, 3, 0, 1, 2]),  # Spot 3: C, G, A, T, C
    ]

    for cycle in range(n_cycles):
        for (cx, cy), sequence in spots:
            channel = sequence[cycle]
            image[:, :, channel, cycle] += makeGaussian(L, fwhm=20, center=(cx, cy))

    return image


@pytest.fixture(scope="session")
def cell_image():
    """2D image with synthetic circular cell-like regions."""
    from scipy.ndimage import gaussian_filter

    L = 256
    image = np.zeros((L, L))

    cell_centers = [(64, 64), (128, 128), (192, 96), (96, 192), (160, 160)]
    cell_radii = [30, 25, 35, 28, 32]

    for (cy, cx), radius in zip(cell_centers, cell_radii):
        y, x = np.ogrid[:L, :L]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        image[mask] = 1.0

    return gaussian_filter(image, sigma=2)


@pytest.fixture(scope="session")
def cell_coords():
    """Expected cell coordinates from synthetic cell image."""
    L = 256
    cell_centers = [(64, 64), (128, 128), (192, 96), (96, 192), (160, 160)]
    cell_radii = [30, 25, 35, 28, 32]

    all_coords = []
    for (cy, cx), radius in zip(cell_centers, cell_radii):
        coords = []
        for y in range(max(0, cy - radius), min(L, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(L, cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    coords.append((x, y))
        all_coords.append(coords)

    return all_coords


@pytest.fixture(scope="session")
def fluo_image(cell_image):
    """Fluorescence image for intensity tests."""
    return cell_image * 100


@pytest.fixture(scope="session")
def image_stack(tmp_path_factory):
    """3D image stack for trajectory tests. Shape: (n_frames, L, L).
    Saves to temp TIF file since student code expects a file path.
    """
    import tifffile

    n_frames = 10
    L = 256

    def make_circle(size, center, radius, intensity=1.0):
        image = np.zeros((size, size))
        y, x = np.ogrid[:size, :size]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        image[mask] = intensity
        return image

    stack = []
    for frame in range(n_frames):
        intensity = 50 + frame * 10
        frame_img = np.zeros((L, L))
        frame_img += make_circle(L, (64, 64), 30, intensity)
        frame_img += make_circle(L, (128, 128), 25, intensity * 1.5)
        frame_img += make_circle(L, (192, 96), 35, intensity * 0.8)
        stack.append(frame_img)

    stack_array = np.array(stack)

    tmp_dir = tmp_path_factory.mktemp("data")
    tif_path = tmp_dir / "test_stack.tif"
    tifffile.imwrite(tif_path, stack_array.astype(np.float32))

    return tif_path
