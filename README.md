# Assignment 3 — Image Analysis

**Course:** g00f3a Computational Biochemistry — Block 2
**Autograder:** GitHub Actions (see the *Actions* tab after pushing)

---

## Learning Objectives

By completing this assignment you will be able to:

1. Detect fluorescent spots in 2D fluorescence images using blob detection algorithms
2. Process multi-channel images and extract nucleotide sequences
3. Track DNA sequences across multiple imaging cycles
4. Segment cells in fluorescence microscopy images
5. Quantify fluorescence intensity from segmented cell regions
6. Analyze time-series fluorescence data to detect cellular dynamics

---

## Prerequisites

| Tool | Install |
|------|---------|
| Git  | <https://git-scm.com> |
| uv   | <https://docs.astral.sh/uv/getting-started/installation/> |
| VSCode (recommended) | <https://code.visualstudio.com> |

---

## Quick Start

```bash
# 1. Clone (GitHub Classroom gives you a personal repo URL)
git clone <your-repo-url>
cd Assignment3_Image_Analysis

# 2. Create the virtual environment and install all dependencies
uv sync

# 3. Run the tests — most will SKIP or FAIL until you implement the functions
uv run pytest

# 4. Open the notebook
uv run marimo edit notebooks/lecture_note_ia_student.marimo.py
```

---

## Project Structure

```
Assignment3_Image_Analysis/
├── .github/workflows/classroom.yml          ← CI autograder (do not modify)
├── notebooks/
│   ├── lecture_note_ia_student.marimo.py    ← YOUR TASK: implement all problems here
│   └── public/                              ← images used by the notebook
├── tests/
│   ├── conftest.py                          ← test fixtures (do not modify)
│   ├── test_1a.py                          ← autograder tests (do not modify)
│   ├── test_1b.py
│   ├── test_1c.py
│   ├── test_2a.py
│   ├── test_2b.py
│   └── test_2c.py
├── pyproject.toml
└── uv.lock
```

---

## Tasks

All tasks are inside `notebooks/lecture_note_ia_student.marimo.py`.
Open it with:

```bash
uv run marimo edit notebooks/lecture_note_ia_student.marimo.py
```

The notebook contains explanations, interactive demos, and `### 📝 Student To-Do` cells
that you must fill in. Below is a brief overview of each exercise.

Note: replace `raise NotImplementedError` with your own code

### Exercise 1 — DNA Sequencing by Synthesis

In this exercise, you will implement functions to detect and track fluorescent spots
that correspond to nucleotides incorporated during DNA sequencing.

| Task | Function | Points |
|------|----------|--------|
| 1A | `find_spots` | 15 |
| 1B | `extract_nucleotides_from_image` | 15 |
| 1C | `full_analysis` | 20 |

#### Problem 1A — Spot Detection

Implement `find_spots(image)`:
- Takes a 2D fluorescence image
- Returns an ndarray of shape (n, 2) containing (x, y) coordinates of detected spots
- Use blob detection algorithms from scikit-image

#### Problem 1B — Nucleotide Extraction

Implement `extract_nucleotides_from_image(image)`:
- Takes a 3D image with 4 channels (A, T, C, G)
- Returns a list of tuples `[(x, y, nucleotide)]`
- Maps each channel to the corresponding Nucleotide enum value

#### Problem 1C — Full Sequence Analysis

Implement `full_analysis(image)`:
- Takes a 4D image stack (x, y, channels, time/cycles)
- Returns a list of tuples `[(x, y, sequence)]` where sequence is a list of nucleotides
- Tracks each spot across all cycles and records its nucleotide sequence

### Exercise 2 — Calcium Imaging Biosensor Analysis

In this exercise, you will analyze calcium imaging data to study cellular dynamics
in response to histamine stimulation.

| Task | Function | Points |
|------|----------|--------|
| 2A | `segment_cells` | 20 |
| 2B | `calculate_cell_fluorescence` | 15 |
| 2C | `fluorescence_trajectory` | 15 |

#### Problem 2A — Cell Segmentation

Implement `segment_cells(single_image)`:
- Takes a 2D fluorescence image
- Returns a list of N sublists, one per cell
- Each sublist contains (x, y) pixel coordinates belonging to that cell
- Use thresholding, morphological operations, and connected component analysis

#### Problem 2B — Fluorescence Intensity Calculation

Implement `calculate_cell_fluorescence(fluo_image, cell_coordinates)`:
- Takes a fluorescence image and the cell coordinate list from 2A
- Returns a list of total fluorescence intensities, one per cell
- Calculate the mean fluorescence within each cell's pixel coordinates

#### Problem 2C — Fluorescence Trajectory Analysis

Implement `fluorescence_trajectory(image)`:
- Takes a full image stack (time series)
- Returns a list of fluorescence trajectories, one per cell
- Plot the trajectories to observe calcium dynamics

---

## Running Tests

```bash
# All tests (skips unimplemented problems automatically)
uv run pytest

# Verbose output
uv run pytest -v

# One exercise at a time
uv run pytest tests/test_1a.py -v
uv run pytest tests/test_1b.py -v
uv run pytest tests/test_1c.py -v
uv run pytest tests/test_2a.py -v
uv run pytest tests/test_2b.py -v
uv run pytest tests/test_2c.py -v

# Stop on first failure
uv run pytest -x
```

---

## Submitting

```bash
# Stage only the notebook you changed
git add notebooks/lecture_note_ia_student.marimo.py

git commit -m "Complete Assignment 3"
git push
```

Then open the **Actions** tab in your GitHub repository to watch the autograder run.
A green checkmark means all tests pass.

---

## Troubleshooting

**Tests skip even though I implemented the function**
Make sure the function name is exactly as specified (e.g. `find_spots`, `segment_cells`).
The autograder finds your code by name. A typo means it won't be found.

**`uv: command not found`**
Install uv following the steps: https://docs.astral.sh/uv/getting-started/installation

**Marimo notebook won't open**
Make sure you are using `uv run marimo edit ...` (not a bare `marimo` command)
so the right virtual environment is used.

**`NotImplementedError` in autograder**
This is expected for any problem you have not yet implemented. Implement the function
and push again.

**Blob detection finds too few/many spots**
Adjust the parameters of the blob detection algorithm (min_sigma, max_sigma, threshold).
The test fixtures use synthetic Gaussian spots, so your algorithm should find them with
appropriate parameters.

**Cell segmentation includes noise**
Use appropriate morphological operations (opening, closing) and filter out small objects.
The minimum cell size parameter is important for removing noise.

**Fluorescence trajectory shows unexpected values**
Make sure you are using the correct pixel access order (row, column = y, x).
Verify that you're calculating the mean, not the sum, of pixel intensities.
