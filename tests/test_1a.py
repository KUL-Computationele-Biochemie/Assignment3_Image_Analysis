from __future__ import annotations
import pytest
import numpy as np
from scipy.stats import qmc


@pytest.fixture
def sample_image():
    """Generates an image with 20 Gaussian spots for testing."""
    L = 1024
    margin = 50
    n_points = 20

    # Poisson disk sampling
    rng = np.random.default_rng(seed=123456789)
    engine = qmc.PoissonDisk(d=2, radius=0.2, rng=rng)
    sample = engine.random(n_points)

    # Scale to [margin, L-margin]
    points = sample * (L - 2 * margin) + margin

    # Gaussian generator
    def makeGaussian(size, fwhm=30, center=None):
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0, y0 = center

        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)

    # Build Image
    image = np.zeros((L, L))
    for p in points:
        image += makeGaussian(size=L, fwhm=30, center=p)

    return image


def make_gaussian_blob(size, center, fwhm=5):
    """Helper method to draw a realistic fluorescent blob on an image (Max intensity 1.0)."""
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = center

    # Create a 2D Gaussian bell curve (DO NOT multiply by 255)
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)


class TestFindSpots:
    def test_function_exists(self, find_spots):
        assert callable(find_spots)

    def test_returns_ndarray(self, find_spots, sample_image):
        assert isinstance(find_spots(sample_image), np.ndarray)

    def test_returns_positive_values(self, find_spots, sample_image):
        result = find_spots(sample_image)
        assert (result >= 0).all()

    def test_finds_single_spot(self, find_spots):
        """Tests if it correctly identifies the (x, y) center of a single blob."""
        image = np.zeros((40, 40))

        # Draw a blob centered at x=25, y=10
        image += make_gaussian_blob(40, center=(25, 10), fwhm=6)

        result = find_spots(image)
        expected = np.array([[25, 10]])

        np.testing.assert_allclose(result, expected, atol=5)

    def test_finds_multiple_spots(self, find_spots):
        """Tests if it handles multiple blobs in the same image."""
        image = np.zeros((50, 50))

        # Draw two separate blobs
        image += make_gaussian_blob(50, center=(12, 8), fwhm=5)  # x=12, y=8
        image += make_gaussian_blob(50, center=(35, 40), fwhm=7)  # x=35, y=40

        result = find_spots(image)

        assert len(result) == 2

        for i, expected_coord in enumerate([[12, 8], [35, 40]]):
            found_match = any(
                abs(result[j, 0] - expected_coord[0]) <= 5
                and abs(result[j, 1] - expected_coord[1]) <= 5
                for j in range(len(result))
            )
            assert found_match, f"No spot found within 5px of {expected_coord}"
