from __future__ import annotations
import pytest
import numpy as np
from enum import Enum


class TestExtractNucleotidesFromImage:
    def test_function_exists(self, extract_nucleotides_from_image):
        assert callable(extract_nucleotides_from_image)

    def test_returns_list(self, extract_nucleotides_from_image, four_channel_image):
        result = extract_nucleotides_from_image(four_channel_image)
        assert isinstance(result, list)

    def test_returns_list_of_tuples(
        self, extract_nucleotides_from_image, four_channel_image
    ):
        result = extract_nucleotides_from_image(four_channel_image)
        assert all(isinstance(t, tuple) and len(t) == 3 for t in result)

    def test_coordinates_are_numeric(
        self, extract_nucleotides_from_image, four_channel_image
    ):
        result = extract_nucleotides_from_image(four_channel_image)
        for x, y, nuc in result:
            assert isinstance(x, (int, float, np.integer, np.floating))
            assert isinstance(y, (int, float, np.integer, np.floating))

    def test_nucleotides_are_enum(
        self, extract_nucleotides_from_image, four_channel_image
    ):
        result = extract_nucleotides_from_image(four_channel_image)
        for x, y, nuc in result:
            assert isinstance(nuc, Enum)

    def test_finds_all_spots(self, extract_nucleotides_from_image, four_channel_image):
        result = extract_nucleotides_from_image(four_channel_image)
        assert len(result) == 4

    def test_channel_order_a_t_c_g(self, extract_nucleotides_from_image):
        """Test that channels are A=0, T=1, C=2, G=3 (correct order).
        This verifies the solution uses the correct channel assignment.
        """
        L = 100
        image = np.zeros((L, L, 4))

        def makeGaussian(size, fwhm=15, center=None):
            x = np.arange(0, size, 1, float)
            y = x[:, np.newaxis]
            if center is None:
                x0, y0 = size // 2, size // 2
            else:
                x0, y0 = center
            return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)

        # Channel 0 = Adenine, Channel 1 = Thymine, Channel 2 = Cytosine, Channel 3 = Guanine
        image[:, :, 0] = makeGaussian(L, center=(25, 25))
        image[:, :, 1] = makeGaussian(L, center=(75, 25))
        image[:, :, 2] = makeGaussian(L, center=(25, 75))
        image[:, :, 3] = makeGaussian(L, center=(75, 75))

        result = extract_nucleotides_from_image(image)

        # Should find exactly 4 spots
        assert len(result) == 4

        # Find the spot near (25, 25) and verify it's Adenine (value=1)
        found_adenine_spot = False
        for x, y, nuc in result:
            if abs(x - 25) <= 5 and abs(y - 25) <= 5:
                assert nuc.value == 1, (
                    f"Channel 0 should be Adenine (value=1), got {nuc.name} (value={nuc.value})"
                )
                found_adenine_spot = True
                break
        assert found_adenine_spot, "Could not find spot near (25, 25)"

    def test_not_wrong_order_t_g_a_c(self, extract_nucleotides_from_image):
        """Test that channels are NOT T=0, G=1, A=2, C=3 (wrong order from hidden text).
        This verifies the solution doesn't use the hidden incorrect channel assignment.
        """
        L = 100
        image = np.zeros((L, L, 4))

        def makeGaussian(size, fwhm=15, center=None):
            x = np.arange(0, size, 1, float)
            y = x[:, np.newaxis]
            if center is None:
                x0, y0 = size // 2, size // 2
            else:
                x0, y0 = center
            return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)

        # Put signal only in channel 0
        image[:, :, 0] = makeGaussian(L, center=(50, 50))

        result = extract_nucleotides_from_image(image)

        # Should find exactly 1 spot
        assert len(result) == 1

        # The spot at (50, 50) should be ADENINE (channel 0 = Adenine with value=1, NOT Thymine with value=4)
        x, y, nuc = result[0]
        assert nuc.value == 1, (
            f"Channel 0 should be Adenine (value=1), got {nuc.name} (value={nuc.value}). "
            "If this shows Thymine, the solution is using the wrong order T=0,G=1,A=2,C=3"
        )
