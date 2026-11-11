"""Preprocessing, normalisation, cosine and pairing logic for the embedder A/B.

Nothing here needs the ONNX weights except the last test, which skips when the
recognition models are absent.
"""

from pathlib import Path

import numpy as np
import pytest

import score_faces_ab
from recognizers import (
    ADAFACE_PREPROCESSING,
    ALIGNED_SIZE,
    ARCFACE_PREPROCESSING,
    align_face,
    l2_normalise,
    load_adaface,
    load_arcface,
    preprocess_blob,
)
from score_faces import cosine_scores, pair_candidates

HERE = Path(__file__).resolve().parent
ARCFACE_PATH = HERE / "model-cache/models/buffalo_l/w600k_r50.onnx"
ADAFACE_PATH = HERE / "model-cache/adaface/adaface_ir_101.onnx"


def constant_crop(values):
    """112x112 BGR crop with one value per channel: (blue, green, red)."""
    crop = np.zeros((ALIGNED_SIZE, ALIGNED_SIZE, 3), dtype=np.uint8)
    crop[:, :, 0], crop[:, :, 1], crop[:, :, 2] = values
    return crop


def scaled(value, mean=127.5, std=127.5):
    """The symmetric normalisation both arms declare: (x - mean) / std."""
    return (value - mean) / std


def test_blob_is_nchw_float32_with_symmetric_range():
    blob = preprocess_blob(constant_crop((0, 127, 255)), "RGB", 127.5, 127.5)
    assert blob.shape == (1, 3, ALIGNED_SIZE, ALIGNED_SIZE)
    assert blob.dtype == np.float32
    # 8-bit values cannot hit the midpoint exactly, so -1 and +1 bracket the range.
    np.testing.assert_allclose(blob[0, :, 0, 0], [scaled(255), scaled(127), scaled(0)],
                               atol=1e-6)


def test_arcface_style_blob_puts_red_first_and_adaface_shares_it():
    crop = constant_crop((0, 127, 255))  # blue 0, green 127, red 255
    rgb = preprocess_blob(crop, "RGB", 127.5, 127.5)
    bgr = preprocess_blob(crop, "BGR", 127.5, 127.5)
    np.testing.assert_allclose(rgb[0, :, 0, 0], [scaled(255), scaled(127), scaled(0)],
                               atol=1e-6)
    np.testing.assert_allclose(bgr[0, :, 0, 0], [scaled(0), scaled(127), scaled(255)],
                               atol=1e-6)
    assert ARCFACE_PREPROCESSING == ADAFACE_PREPROCESSING


def test_blob_rejects_wrong_shape_dtype_and_channel_order():
    with pytest.raises(ValueError, match="crop"):
        preprocess_blob(np.zeros((96, 96, 3), dtype=np.uint8), "RGB", 127.5, 127.5)
    with pytest.raises(ValueError, match="uint8"):
        preprocess_blob(np.zeros((ALIGNED_SIZE, ALIGNED_SIZE, 3), dtype=np.float32),
                        "RGB", 127.5, 127.5)
    with pytest.raises(ValueError, match="channel order"):
        preprocess_blob(constant_crop((0, 127, 255)), "YUV", 127.5, 127.5)


def test_normalisation_makes_unit_vectors_and_rejects_degenerate_ones():
    vector = l2_normalise(np.arange(1, 513, dtype=np.float32))
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6
    with pytest.raises(ValueError, match="degenerate"):
        l2_normalise(np.zeros(512, dtype=np.float32))
    with pytest.raises(ValueError, match="512"):
        l2_normalise(np.ones(256, dtype=np.float32))


def test_normalised_cosine_ignores_embedding_scale():
    raw = np.arange(1, 513, dtype=np.float32)
    assert abs(float(l2_normalise(raw) @ l2_normalise(3 * raw)) - 1.0) < 1e-6


def test_rank1_margin_is_assigned_minus_best_other_reference():
    scores = np.array([[0.9, 0.2], [0.5, 0.1]])
    assert score_faces_ab.rank1_margin(scores, 0, 0) == pytest.approx(0.4)
    # A negative margin says another character's portrait matches this face better.
    assert score_faces_ab.rank1_margin(scores, 1, 0) == pytest.approx(-0.4)


def test_rank1_margin_is_none_with_a_single_reference():
    assert score_faces_ab.rank1_margin(np.array([[0.9]]), 0, 0) is None


def test_pairing_and_cosine_come_from_the_frozen_baseline_module():
    assert score_faces_ab.cosine_scores is cosine_scores
    assert score_faces_ab.pair_candidates is pair_candidates
    scores = np.array([[0.9, 0.8], [0.85, 0.1]])
    assert pair_candidates(scores) == [1, 0]


def test_alignment_uses_landmarks_and_returns_a_112_crop():
    image = np.full((256, 256, 3), 200, dtype=np.uint8)
    image[80:160, 90:150] = 40
    landmarks = np.array([[100, 100], [130, 100], [115, 120], [103, 140], [127, 140]],
                         dtype=np.float32)
    crop = align_face(image, landmarks)
    assert crop.shape == (ALIGNED_SIZE, ALIGNED_SIZE, 3)
    assert crop.dtype == np.uint8
    moved = align_face(image, landmarks + np.array([20, 0], dtype=np.float32))
    assert not np.array_equal(crop, moved)


def test_ab_cli_defaults_to_its_own_output_directory():
    args = score_faces_ab.parse_args([])
    assert args.output == HERE / "results_adaface"
    assert args.images == score_faces_ab.PROJECT / "output/images"
    assert args.sweep_preprocessing is False


def test_ab_cli_keeps_an_explicit_output_and_accepts_the_sweep_flag(tmp_path):
    args = score_faces_ab.parse_args(["--output", str(tmp_path), "--sweep-preprocessing"])
    assert args.output == tmp_path
    assert args.sweep_preprocessing is True


@pytest.mark.skipif(not (ARCFACE_PATH.exists() and ADAFACE_PATH.exists()),
                    reason="recognition weights not present")
def test_both_arms_embed_the_same_crop_to_unit_vectors():
    crop = np.full((ALIGNED_SIZE, ALIGNED_SIZE, 3), 128, dtype=np.uint8)
    arms = {"arcface": load_arcface(ARCFACE_PATH), "adaface": load_adaface(ADAFACE_PATH)}
    embeddings = {name: embedder.embed(crop) for name, embedder in arms.items()}
    for embedding in embeddings.values():
        assert embedding.shape == (512,)
        assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-5
    assert not np.allclose(embeddings["arcface"], embeddings["adaface"])
    # AdaFace already normalises inside the graph; ArcFace does not.
    assert abs(float(np.linalg.norm(arms["adaface"].forward(crop))) - 1.0) < 1e-4
    assert float(np.linalg.norm(arms["arcface"].forward(crop))) > 2.0
