from __future__ import annotations
import pytest
import numpy as np
from enum import Enum


class TestFullAnalysis:
    def test_function_exists(self, full_analysis):
        assert callable(full_analysis)

    def test_returns_list(self, full_analysis, four_dim_image):
        result = full_analysis(four_dim_image)
        assert isinstance(result, list)

    def test_returns_list_of_tuples(self, full_analysis, four_dim_image):
        result = full_analysis(four_dim_image)
        assert all(isinstance(t, tuple) and len(t) == 3 for t in result)

    def test_sequence_is_list(self, full_analysis, four_dim_image):
        result = full_analysis(four_dim_image)
        for x, y, seq in result:
            assert isinstance(seq, list)

    def test_sequence_contains_nucleotides(self, full_analysis, four_dim_image):
        result = full_analysis(four_dim_image)
        for x, y, seq in result:
            assert all(isinstance(n, Enum) for n in seq)

    def test_sequence_length_matches_cycles(self, full_analysis, four_dim_image):
        result = full_analysis(four_dim_image)
        n_cycles = four_dim_image.shape[3]
        for x, y, seq in result:
            assert len(seq) == n_cycles

    def test_finds_all_dna_fragments(self, full_analysis, four_dim_image):
        result = full_analysis(four_dim_image)
        assert len(result) == 3

    def test_returns_tuples_not_just_sequences(self, full_analysis, four_dim_image):
        """Verify function returns [(x, y, sequence)] tuples, NOT just sequences.
        This checks against the hidden text behavior which says to return just sequences.
        """
        result = full_analysis(four_dim_image)

        assert isinstance(result, list)
        assert all(isinstance(t, tuple) for t in result)

        for item in result:
            assert len(item) == 3, f"Each tuple should have 3 elements, got {len(item)}"
            x, y, seq = item
            assert isinstance(x, (int, float, np.integer, np.floating)), (
                f"x should be numeric, got {type(x)}"
            )
            assert isinstance(y, (int, float, np.integer, np.floating)), (
                f"y should be numeric, got {type(y)}"
            )
            assert isinstance(seq, list), f"sequence should be a list, got {type(seq)}"

    def test_sequences_match_expected(self, full_analysis, four_dim_image):
        """Verify that sequences match the expected patterns.
        Channel order: A=0, T=1, C=2, G=3
        Spot 1 at (50, 50): channels [0, 1, 2, 3, 0] = A(1), T(4), C(3), G(2), A(1)
        Spot 2 at (100, 80): channels [1, 2, 3, 0, 1] = T(4), C(3), G(2), A(1), T(4)
        Spot 3 at (150, 120): channels [2, 3, 0, 1, 2] = C(3), G(2), A(1), T(4), C(3)
        """
        expected_sequences = [
            ((50, 50), [1, 4, 3, 2, 1]),  # A, T, C, G, A
            ((100, 80), [4, 3, 2, 1, 4]),  # T, C, G, A, T
            ((150, 120), [3, 2, 1, 4, 3]),  # C, G, A, T, C
        ]

        result = full_analysis(four_dim_image)
        assert len(result) == 3

        for expected_pos, expected_seq_values in expected_sequences:
            expected_x, expected_y = expected_pos
            found = False
            for x, y, seq in result:
                if abs(x - expected_x) <= 5 and abs(y - expected_y) <= 5:
                    seq_values = [n.value for n in seq]
                    assert seq_values == expected_seq_values, (
                        f"Sequence at ({x}, {y}) does not match expected. "
                        f"Got {[n.name for n in seq]} ({seq_values}), "
                        f"expected {[Nucleotide(v).name for v in expected_seq_values]} ({expected_seq_values})"
                    )
                    found = True
                    break
            assert found, (
                f"No sequence found near expected position ({expected_x}, {expected_y})"
            )

    def test_positions_within_tolerance(self, full_analysis, four_dim_image):
        """Verify that detected positions are within 5 pixels of expected."""
        result = full_analysis(four_dim_image)

        expected_positions = [(50, 50), (100, 80), (150, 120)]

        for expected_x, expected_y in expected_positions:
            found = False
            for x, y, seq in result:
                if abs(x - expected_x) <= 5 and abs(y - expected_y) <= 5:
                    found = True
                    break
            assert found, (
                f"No spot found within 5px of expected position ({expected_x}, {expected_y})"
            )
