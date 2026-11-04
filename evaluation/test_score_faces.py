"""Meaningful edge cases for scoring and one-to-one candidate association."""

import numpy as np

from score_faces import cosine_scores, pair_candidates


def test_cosine_is_scale_independent_and_handles_opposite_faces():
    scores = cosine_scores([[2, 0], [0, 3]], [[4, 0], [-1, 0], [0, 7]])
    np.testing.assert_allclose(scores, [[1, -1, 0], [0, 0, 1]])


def test_no_faces_leaves_every_expected_character_unassigned():
    scores = cosine_scores([[1, 0], [0, 1]], [])
    assert scores.shape == (2, 0)
    assert pair_candidates(scores) == [None, None]


def test_global_assignment_does_not_reuse_a_face_or_greedily_steal_it():
    scores = np.array([[0.9, 0.8], [0.85, 0.1]])
    assert pair_candidates(scores) == [1, 0]


def test_fewer_faces_than_people_preserves_unassigned_character():
    scores = np.array([[0.9, 0.1], [0.1, 0.2], [0.0, 0.8]])
    assert pair_candidates(scores) == [0, None, 1]


def test_more_faces_than_people_does_not_force_every_face_into_a_character():
    scores = np.array([[0.1, 0.9, 0.2], [0.2, 0.1, 0.8]])
    assert pair_candidates(scores) == [1, 2]


def test_weak_pairing_is_still_only_a_candidate_not_a_threshold_decision():
    assert pair_candidates(np.array([[-0.2]])) == [0]


def test_no_expected_people():
    assert pair_candidates(np.empty((0, 2))) == []
