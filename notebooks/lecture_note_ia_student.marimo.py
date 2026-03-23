import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import qmc
    from enum import Enum
    import math
    import tifffile

    return Enum, np, plt, qmc, tifffile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grading Overview

    | Problem | Task | Points |
    |---------|------|--------|
    | **1A** | Find Spots (`find_spots`) | 15 |
    | **1B** | Extract Nucleotides (`extract_nucleotides_from_image`) | 15 |
    | **1C** | Full Analysis (`full_analysis`) | 20 |
    | **2A** | Segment Cells (`segment_cells`) | 20 |
    | **2B** | Cell Fluorescence (`calculate_cell_fluorescence`) | 15 |
    | **2C** | Fluorescence Trajectory (`fluorescence_trajectory`) | 15 |
    | | **Total** | **100** |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    check_score_btn = mo.ui.button(
        label="Check my score",
        value=0,
        on_click=lambda v: v + 1,
    )
    mo.md(f"""
    ### Score Checker

    Click the button to run the autograder locally and see your current score:

    {check_score_btn}
    """)
    return (check_score_btn,)


@app.cell(hide_code=True)
def _(check_score_btn, mo):
    import subprocess as _subprocess
    from pathlib import Path as _Path

    mo.stop(
        check_score_btn.value == 0,
        mo.md("_Click the button above to run the autograder._"),
    )

    try:
        _repo_root = _Path(mo.app_meta().filename).resolve().parent.parent
    except Exception:
        import os as _os

        _repo_root = _Path(_os.getcwd())
    while not (_repo_root / "tests").exists() and _repo_root != _repo_root.parent:
        _repo_root = _repo_root.parent

    _GRADING = [
        ("1A: Find Spots", "tests/test_1a.py", 15),
        ("1B: Extract Nucleotides", "tests/test_1b.py", 15),
        ("1C: Full Analysis", "tests/test_1c.py", 20),
        ("2A: Segment Cells", "tests/test_2a.py", 20),
        ("2B: Cell Fluorescence", "tests/test_2b.py", 15),
        ("2C: Fluorescence Trajectory", "tests/test_2c.py", 15),
    ]

    _rows = []
    _total_earned = 0
    for _name, _test_file, _pts in _GRADING:
        _proc = _subprocess.run(
            ["uv", "run", "pytest", _test_file, "-q", "--tb=no"],
            capture_output=True,
            text=True,
            cwd=str(_repo_root),
        )
        _passed = _proc.returncode == 0
        _earned = _pts if _passed else 0
        _total_earned += _earned
        _icon = "✅" if _passed else "❌"
        _rows.append(f"| {_icon} | {_name} | {_earned} / {_pts} |")

    _total_max = sum(_p for _, _, _p in _GRADING)
    _table = "\n".join(_rows)

    mo.md(f"""
    ## Your current score: {_total_earned} / {_total_max}

    | Status | Problem | Points |
    |--------|---------|--------|
    {_table}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 1:
    Many methods exist to obtain the sequences of DNA fragments. One of these is known as
    ’sequencing by synthesis’, which relies on the detection of fluorescently labelled nucleotides
    as they are incorporated into a growing DNA strand. The fluorescence signal is detected
    using a microscope and the resulting images are analyzed to determine the sequence of the
    DNA. This figure shows the concept:
    <img src="https://data-science-sequencing.github.io/Win2018/assets/lecture2/Figure5_IlluminaInfoCollection.png" width=50%>

    In a nutshell, the following process takes place:
    1. The DNA fragments to be sequenced are attached to a cover glass and placed on a
    fluorescence microscope. Each DNA fragment (should) end up at a unique spatial
    location on the glass. The sample is placed in a solution that contains the necessary
    enzymes and reagents for DNA synthesis.
    2. Labeled nucleotides are added to the solution. The correct nucleotide is incorporated
    into the DNA fragments, causing it to become fluorescent. The color of the fluorescence
    indicates which nucleotide was incorporated (e.g., A, T, C, or G). The nucleotides have
    also been chemically modified so that only a single nucleotide can be incorporated.
    The enzyme cannot add a second nucleotide.
    3. The microscope acquires images of the sample. Four fluorescence images are recorded,
    one showing the fluorescence from A, one showing T, one showing C, and the last showing G. (We do not see the nucleotides directly, of course. We see the fluorescence from the labels attached to the nucleotides.)
    4. The images are analyzed to determine which nucleotide was incorporated. Is there
    fluorescence in a given location, and what color is it?
    5. The attached nucleotides are chemically treated so that a subsequent nucleotide can
    be incorporated in the next cycle.
    6. Steps 2-5 are repeated until the sequencing has progressed sufficiently far.

    In a real experiment, step 4 does not need to be done during the measurement. It can be
    done later as part of the data analysis.
    Your job is to develop the data processing software. Your program will be given a fourdimensional
    NDArray. The first two dimensions correspond to the spatial dimensions of the
    image (i.e., the x and y coordinates). The third dimension corresponds to the four different
    fluorescence channels (i.e., A, T, C, and G). The fourth dimension corresponds to the different
    cycles of the sequencing process. Your program should analyze this data and calculate the
    sequences of the DNA fragments.
    The data itself is greatly simplified. We assume that the fluorophores are very bright,
    emit only in a single channel, that there is no background fluorescence, and that there is
    no noise in the measurement. In a real experiment these issues would cause significant
    complications, but we will ignore them for the sake of simplicity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 1A
    Given a fluorescence image, we will need to determine if there are fluorescent spots
    present, and if so, where they are located. Write a function that takes a two-dimensional
    array (a single fluorescence image) and the returns a list of all <span style="font-size: 0.1px">-</span>𝑥, <span style="font-size: 0.1px">-</span>𝑦 values where fluorescence spots have been found <span style="font-size: 0.1px"> but negative (* -1) </span> in the form of a ndarray with the following shape (&lt;number of points&gt;, 2)

    Complete the function <code>find_spots</code> to match the required functionality

    We have created some sample data for you to examine <code>image</code> is a 2D NDArray that contains fluorescent spots. Its rendered in the cell below.

    >### Hints
    >There are many algorithms you can use. If you use a specific one from a library make sure to import it inside the function body.
    >Example:
    >```python
    >def find_spots(image):
    >    import your_algorithm_here
    >```
    >We recommend using either:
    >
    > - [`Blob detection algorithms`](https://scikit-image.org/docs/0.23.x/auto_examples/features_detection/plot_blob.html) from scikit-image .
    > - Threshold image and [`connected components`](https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.label) from scikit-image .
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, qmc):
    # Parameters
    L = 1024  # Size
    margin = 50  # Margin from border
    n_points = 20  # Number of points

    # Poisson disk sampling
    rng = np.random.default_rng(seed=67)
    engine = qmc.PoissonDisk(d=2, radius=0.2, rng=rng)

    sample = engine.random(n_points)

    # scale to [margin, L-margin]
    points = sample * (L - 2 * margin) + margin

    # Gausian
    def makeGaussian(size, fwhm=30, center=None):
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0, y0 = center

        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)

    # -----------------------
    # Image
    # -----------------------
    image = np.zeros((L, L))

    for p in points:
        image += makeGaussian(size=L, fwhm=30, center=p)

    # -----------------------
    # Show
    # -----------------------
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    # plt.show()
    return L, image, makeGaussian, points


@app.function
def find_spots(image):
    # Student TODO
    raise NotImplementedError


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell below can be used to test your function. If your function works you should see red dots on top of each spot of the image
    """)
    return


@app.cell(hide_code=True)
def _(image, plt):
    _imagepoints = find_spots(image)
    print(_imagepoints.shape)
    x = _imagepoints[:, 0]
    y = _imagepoints[:, 1]
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="grey")
    ax.scatter(x, y, label="your fit", color="red")
    ax.legend()
    ax.set_axis_off()
    ax.set_title("Original Image and found points")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 1B
    Next, we will need to apply this function to all four images acquired in a given
    cycle. Write a function that takes a three-dimensional NDArray, where the first
    two dimensions correspond to the spatial dimensions of the image and the
    third dimension corresponds to the four different fluorescence channels (in
    the order <span style="font-size: 0.1px"> T, G, A, C. NOT</span>  A, T, C, G). The result of this function should be a list of tuples of the
    form <span style="font-size: 0.1px"> [(nucleotide,𝑦,𝑥)]</span>[(𝑥, 𝑦, nucleotide)].

    Complete the function <code>extract_nucleotides_from_image</code> to match the required functionality.
    Use the ```Nucleotide``` enum that we have created for you like this.

    ```python
    someNucleotide = Nucleotide.ADENINE
    otherNucleotide = Nucleotide.GUANINE
    ...
    ```

    We have created some sample data for you to examine; <code>four_channel_image</code> is a 3D NDArray that contains fluorescent spots. Its rendered in the cell below.
    """)
    return


@app.cell(hide_code=True)
def _(L, makeGaussian, np, plt, points):
    # Parameters

    four_channel_image = np.zeros((L, L, 4))
    random_nucleotides = np.random.randint(low=0, high=4, size=(20))

    for point, n in zip(points, random_nucleotides):
        four_channel_image[:, :, n] += makeGaussian(size=L, fwhm=30, center=point)

    # # -----------------------
    # # Show
    # # -----------------------

    def channels_to_rgb(img):
        """Convert 4-channel image to RGB using R,G,Y,B mapping."""
        L = img.shape[0]
        rgb = np.zeros((L, L, 3))

        # Map channels to colors
        rgb[:, :, 0] += img[:, :, 0]  # Red
        rgb[:, :, 1] += img[:, :, 1]  # Green
        rgb[:, :, 0] += img[:, :, 2]  # Yellow = Red + Green
        rgb[:, :, 1] += img[:, :, 2]
        rgb[:, :, 2] += img[:, :, 3]  # Blue

        # Clip values to [0,1] for display
        rgb = np.clip(rgb, 0, 1)
        return rgb

    fig_4_channels, ax_4_channels = plt.subplots(ncols=4, nrows=1)
    for channel in range(4):
        ax_4_channels[channel].imshow(four_channel_image[:, :, channel], cmap="grey")
        ax_4_channels[channel].set_axis_off()
        ax_4_channels[channel].set_title(f"channel:{channel}")
    plt.show()
    return (four_channel_image,)


@app.cell
def _(Enum):
    # We have created this helper for you
    class Nucleotide(Enum):
        ADENINE = 1
        GUANINE = 2
        CYTOSINE = 3
        THYMINE = 4

    someNucleotide = Nucleotide.ADENINE

    print(someNucleotide)
    return


@app.function
def extract_nucleotides_from_image(image):
    # Student TODO
    raise NotImplementedError


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell below can be used to test your function. If your function works you should see in the cell output a long list of tuples containing the position and the nucleotide
    """)
    return


@app.cell
def _(four_channel_image):
    nucs = extract_nucleotides_from_image(four_channel_image)
    nucs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 1C
    Now write a function that does the full analysis. Your function should take the full fourdimensional
    NDArray as input and return <span style="font-size: 0.1px"> not</span> a list of tuples of the form [(𝑥, 𝑦, sequence)] <span style="font-size: 0.1px"> but all the sequences combined, and ignore x and y</span>,
    where 𝑥 and 𝑦 are the spatial coordinates of the DNA fragments, and ’sequence’ is a
    list with the sequence of nucleotides that were incorporated at that location over the
    different cycles.

    Complete the function <code>full_analysis</code> to match the required functionality

    We have created some sample data for you to examine; <code>four_dimensions_four_channel_image</code> is 4D NDArray that contains fluorescent spots. Its rendered in the cell below.
    """)
    return


@app.cell(hide_code=True)
def _(L, makeGaussian, np, points):
    four_dimensions_four_channel_image = np.zeros((L, L, 4, 20))
    for t in range(20):
        np.random.seed(t)
        t_random_nucleotides = np.random.randint(low=0, high=4, size=(20))
        for _point, t_random_nucleotide in zip(points, t_random_nucleotides):
            four_dimensions_four_channel_image[:, :, t_random_nucleotide, t] += (
                makeGaussian(size=L, fwhm=30, center=_point)
            )
    return (four_dimensions_four_channel_image,)


@app.cell(hide_code=True)
def _(mo):
    slider = mo.ui.slider(start=0, stop=19, label="Frame", value=1)
    return (slider,)


@app.cell(hide_code=True)
def _(four_dimensions_four_channel_image, mo, plt, slider):
    fig_4_channels_t, ax_4_channels_t = plt.subplots(ncols=4, nrows=1)
    for channel_t in range(4):
        ax_4_channels_t[channel_t].imshow(
            four_dimensions_four_channel_image[:, :, channel_t, slider.value],
            cmap="grey",
        )
        ax_4_channels_t[channel_t].set_axis_off()
        ax_4_channels_t[channel_t].set_title(f"channel:{channel_t}")
    fig_4_channels_t.suptitle(f"Frame = {slider.value}")
    plt.tight_layout()

    fax = mo.ui.matplotlib(plt.gca())
    mo.vstack([slider, fax])
    return


@app.function
def full_analysis(image):
    # Student TODO
    raise NotImplementedError


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell below can be used to test your function. If your function works you should see in the cell output a long list with the sequences found.
    Warning⚠️ : This cell will take some time to perform the whole analysis. It's deactivated by default. Click the (···) button of the cell below to enable code execution. Disable it once the code is fully executed.
    """)
    return


@app.cell(disabled=True)
def _(four_dimensions_four_channel_image):
    analysis = full_analysis(image=four_dimensions_four_channel_image)

    def print_sequences(sequences):
        nucleotide_map = {
            "Nucleotide.ADENINE": "A",
            "Nucleotide.THYMINE": "T",
            "Nucleotide.CYTOSINE": "C",
            "Nucleotide.GUANINE": "G",
        }

        print(f"Found {len(sequences)} sequences:")
        for i, (x, y, seq) in enumerate(sequences, start=1):
            seq_str = "".join(nucleotide_map.get(str(n), "?") for n in seq)
            print(f"    seq{i} ({x:.1f}, {y:.1f}): {seq_str}")

    print_sequences(analysis)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 2:
    Biosensors are fluorescent molecules that change their fluorescence when something in
    their environment changes. Biosensors for Ca<sup>2+</sup>, for example, become brighter when the
    local concentration of calcium increases, and become dimmer when it decreases. Because
    Ca<sup>2+</sup> is an important signaling molecule in cells, these biosensors are widely used to monitor
    cellular activity. The fluorescence of the biosensor is typically measured using a microscope,
    and the resulting images are analyzed to determine the concentration of Ca<sup>2+</sup> in (different
    parts of) the cell.
    You have used a microscope to acquire fluorescence images on live cells expressing a
    biosensor for this ion. During the experiment, you have added histamine in order to trigger
    calcium signals in the cells. Your goal is to determine the biosensor signal as a function of
    time, for each cell individually.

    The raw data is provided in the file <code>calcium_imaging_data.tif</code>. This file contains multiple fluorescence images aquired over time. Load the data into Python and look at the images visually. Change the colorscale setting to make it easier to see the cells and the calcium dynamics. Yo
    """)
    return


@app.cell
def _(tifffile):
    path = "notebooks/public/calcium_imaging_data.tif"
    image_stack = tifffile.imread(path)
    return (image_stack,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 2A:
    We need to figure out where the different cells are in the image. Write a function that takes a single fluorescence image and that returns a <span style="font-size: 0.1px">a numpy array or dict but not a</span>list of 𝑁 sublists, one per cell, where each sublist contains the (𝑥, 𝑦) <span style="font-size: 0.1px">trasposed,</span> pixel coordinates belonging to that cell (in other words, the inner lists contain the coordinates of all pixels in the image 'covered' by a cell).

    >### Hints
    >There are several steps involved in cell segmentation cell segmentation algorithms. Skimage and Scipy have functions for most of these steps
    >Example:
    """)
    return


@app.cell
def _(np):
    def segment_cells(single_image: np.ndarray):
        # Student TODO
        raise NotImplementedError

    return (segment_cells,)


@app.cell
def _(image_stack, segment_cells):
    segmentation = segment_cells(image_stack[0])
    return (segmentation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 2B:
    Write a function that takes a fluorescence image and the list from (b), and returns a list with the <span style="font-size: 0.1px">sum instead of</span>average fluorescence intensity for every cell in that image.
    """)
    return


@app.cell
def _(np):
    def calculate_cell_fluorescence(fluo_image: np.ndarray, cell_coordinates: list):
        # Student TODO
        raise NotImplementedError

    return (calculate_cell_fluorescence,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If your functions works below you shold see a list with the intensities
    """)
    return


@app.cell
def _(calculate_cell_fluorescence, image_stack, segmentation):
    intensities = calculate_cell_fluorescence(image_stack[0], segmentation)
    print(intensities)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 2C:
    Now write a function that takes<span style="font-size: 0.1px">image stack extracted</span> the image path and that calculate the full fluorescence trajectory. Plot these signals visually. Do you see any interesting dynamics? Can you identify the moment when histamine was added?
    """)
    return


@app.function
def fluorescence_trajectory(image):
    # Student TODO
    raise NotImplementedError


@app.cell
def _():
    # Plot trajectories here

    return


if __name__ == "__main__":
    app.run()
