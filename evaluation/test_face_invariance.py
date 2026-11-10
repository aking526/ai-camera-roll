"""Checks for the identity-preserving perturbation grid and the derived floor.

Pure-logic tests run without model weights. Anything that needs the ArcFace or
SCRFD weights is skipped unless both verified files are present locally.
"""

import hashlib
import json

import cv2
import numpy as np
import pytest

from face_invariance import (
    BASE_SEED,
    FACE_SCALE_INTEROCULAR,
    HERE,
    OCCLUSION_FILL,
    PROJECT,
    aggregate,
    apply_color_temperature,
    apply_gain_offset,
    apply_gamma,
    apply_gaussian_blur,
    apply_jpeg,
    apply_lighting_gradient,
    apply_occlusion,
    apply_perturbation,
    band_patch,
    between_identity_gap,
    check_iteration01,
    classify_assignments,
    color_temperature_gains,
    cosine,
    downscale_restore,
    embed_crop,
    family_curve,
    multipass_drops,
    occluded_patch,
    parse_args,
    parse_face_scores,
    perspective_warp,
    perturbation_grid,
    project_corners,
    realistic_family_maxima,
    select_floor,
    sha256,
    signal_to_noise,
    stable_rng,
    summarise_level,
)

MODELS = HERE / "model-cache/models/buffalo_l"
WEIGHTS_PRESENT = all((MODELS / name).is_file()
                      for name in ("det_10g.onnx", "w600k_r50.onnx"))
needs_weights = pytest.mark.skipif(not WEIGHTS_PRESENT, reason="verified insightface weights absent")

TEMPLATE_KPS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]])


def synthetic_image(seed=0, size=64):
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 255, size, dtype=float)
    image = np.zeros((size, size, 3), np.uint8)
    for channel in range(3):
        image[:, :, channel] = np.clip(ramp[:, None] * (0.5 + 0.2 * channel)
                                       + rng.normal(0, 4, (size, size)), 0, 255)
    return image


def face_like_crop():
    """A 112x112 stand-in with a bright eye band and a darker mouth band."""
    crop = np.full((112, 112, 3), 90, np.uint8)
    cv2.circle(crop, (38, 52), 8, (230, 230, 230), -1)
    cv2.circle(crop, (74, 52), 8, (230, 230, 230), -1)
    cv2.line(crop, (42, 92), (70, 92), (40, 40, 40), 6)
    return crop


def grid_entry(family, level):
    for entry in perturbation_grid():
        if entry["family"] == family and entry["level"] == level:
            return entry
    raise AssertionError(f"missing grid entry {family}/{level}")


# ---------------------------------------------------------------- cosine


def test_cosine_matches_identical_orthogonal_and_opposite_vectors():
    assert cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert cosine([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude_and_accepts_two_dimensional_embeddings():
    assert cosine(np.array([[3.0, 4.0]]), np.array([[6.0, 8.0]])) == pytest.approx(1.0)


# ---------------------------------------------------------------- pose


def test_zero_pose_leaves_the_image_pixel_identical():
    image = synthetic_image()
    warped, projected = perspective_warp(image, 0, 0, 0)
    assert warped.shape == image.shape
    assert np.array_equal(warped, image)
    np.testing.assert_allclose(projected[:, 0], [0, image.shape[1] - 1,
                                                 image.shape[1] - 1, 0], atol=1e-9)


def test_pose_homography_shrinks_the_projected_quad_monotonically():
    focal = 3.0 * 512
    corners, projected = project_corners(512, 512, 0, 0, 0, focal)
    plain = cv2.contourArea(projected.astype(np.float32))
    areas = [cv2.contourArea(project_corners(512, 512, angle, 0, 0, focal)[1].astype(np.float32))
             for angle in (10, 20, 30, 40, 50)]
    assert all(area < plain for area in areas)
    assert areas == sorted(areas, reverse=True)


def test_yaw_sign_moves_the_projected_quad_to_opposite_sides():
    focal = 3.0 * 512
    _, left = project_corners(512, 512, 20, 0, 0, focal)
    _, right = project_corners(512, 512, -20, 0, 0, focal)
    assert left[:, 0].mean() > right[:, 0].mean()
    assert np.allclose(left[:, 1].mean(), right[:, 1].mean(), atol=1e-6)


def test_each_pose_axis_compresses_or_rotates_a_different_extent():
    focal = 3.0 * 512
    _, yaw = project_corners(512, 512, 20, 0, 0, focal)
    _, pitch = project_corners(512, 512, 0, 20, 0, focal)
    _, roll = project_corners(512, 512, 0, 0, 20, focal)
    assert np.ptp(yaw[:, 0]) < 511.0
    assert np.ptp(pitch[:, 1]) < 511.0
    # An in-plane roll keeps the area and grows the axis-aligned bounding box.
    assert cv2.contourArea(roll.astype(np.float32)) == pytest.approx(511.0 ** 2, rel=1e-6)
    assert np.ptp(roll[:, 0]) > 511.0 and np.ptp(roll[:, 1]) > 511.0


@pytest.mark.parametrize("side,focal_px", [(1024, 0.7 * 1024), (448, 0.7 * 448),
                                           (112, 0.7 * 1024 * 112 / 266),
                                           (112, 0.7 * 1024 * 112 / 130)])
def test_every_grid_pose_stays_in_front_of_the_camera(side, focal_px):
    for entry in perturbation_grid():
        if entry["family"] != "pose":
            continue
        angle = entry["params"]["angle_deg"]
        project_corners(side, side, angle, 0, 0, focal_px)
        project_corners(side, side, 0, angle, 0, focal_px)
        project_corners(side, side, 0, 0, angle, focal_px)


def test_pose_rotates_about_the_named_centre():
    # A corner that lies on the rotation axis does not move.
    _, on_axis = project_corners(512, 512, 25, 15, 20, 0.7 * 512, (0.0, 0.0))
    np.testing.assert_allclose(on_axis[0], [0.0, 0.0], atol=1e-9)
    # Turning the face about its own centre is not the same warp as turning the
    # whole photo about the image centre.
    _, about_face = project_corners(512, 512, 0, 30, 0, 0.7 * 512, (300.0, 200.0))
    _, about_image = project_corners(512, 512, 0, 30, 0, 0.7 * 512)
    assert not np.allclose(about_face, about_image)


def test_pose_focal_length_keeps_the_face_ratio_across_regimes():
    from face_invariance import ALIGNED_CROP_SIZE, ALIGNED_TEMPLATE_KPS, pose_focal_px

    record = {"image_width": 1024, "bbox": [100.0, 100.0, 334.0, 400.0]}
    photo_focal = pose_focal_px("end_to_end", record)
    crop_focal = pose_focal_px("embed_only", record)
    assert photo_focal == pytest.approx(0.7 * 1024)
    # Same focal-to-face-width ratio, so the face turns by the same amount.
    assert crop_focal / ALIGNED_CROP_SIZE == pytest.approx(photo_focal / 234.0)
    assert ALIGNED_TEMPLATE_KPS.shape == (5, 2)


def test_pose_warp_preserves_three_channels_and_a_visible_face_region():
    image = cv2.resize(face_like_crop(), (512, 512), interpolation=cv2.INTER_NEAREST)
    warped, _ = perspective_warp(image, 30, 0, 0)
    assert warped.ndim == 3 and warped.shape[2] == 3
    assert int(warped.mean()) > 0


# ---------------------------------------------------------------- illumination


def test_gamma_keeps_the_endpoints_and_brightens_below_one():
    image = np.array([[[0, 128, 255]]], np.uint8)
    bright = apply_gamma(image, 0.5)
    dark = apply_gamma(image, 2.0)
    assert bright[0, 0, 0] == 0 and bright[0, 0, 2] == 255
    assert bright[0, 0, 1] > image[0, 0, 1] > dark[0, 0, 1]


def test_gain_offset_is_clipped_and_centred_on_mid_grey():
    image = np.array([[[10, 128, 250]]], np.uint8)
    np.testing.assert_array_equal(apply_gain_offset(image, 1.0, 0.0), image)
    brighter = apply_gain_offset(image, 1.0, 40.0)
    np.testing.assert_array_equal(brighter, [[[50, 168, 255]]])
    assert apply_gain_offset(image, 1.0, -80.0)[0, 0, 0] == 0


def test_colour_temperature_warm_boosts_red_and_cool_boosts_blue():
    warm_red, _, warm_blue = color_temperature_gains(3000)
    cool_red, _, cool_blue = color_temperature_gains(9000)
    assert warm_red > 1.0 > warm_blue
    assert cool_blue > 1.0 > cool_red
    assert color_temperature_gains(3000)[0] > color_temperature_gains(4500)[0]


def test_colour_temperature_keeps_pixels_in_range_and_green_fixed():
    image = np.full((4, 4, 3), 250, np.uint8)
    warm = apply_color_temperature(image, 3000)
    assert warm.dtype == np.uint8
    assert warm.max() <= 255
    assert warm[0, 0, 1] == 250
    assert warm[0, 0, 0] == 255


def test_lighting_gradient_spans_exactly_the_requested_range():
    image = np.full((32, 32, 3), 100, np.uint8)
    shaded = apply_lighting_gradient(image, 0.5, 0.0).astype(float)
    assert shaded[:, :, 0].min() == pytest.approx(50, abs=1)
    assert shaded[:, :, 0].max() == pytest.approx(150, abs=1)
    assert shaded[0, 0, 0] < shaded[0, -1, 0]


# ---------------------------------------------------------------- resolution


def test_downscale_restore_keeps_the_original_shape():
    image = synthetic_image(size=96)
    assert downscale_restore(image, 40).shape == image.shape
    assert downscale_restore(image, 200).shape == image.shape


def test_downscale_to_the_native_size_changes_nothing():
    image = synthetic_image(size=64)
    np.testing.assert_array_equal(downscale_restore(image, 64), image)


def test_lower_resolution_moves_further_from_the_original():
    image = synthetic_image(size=128)
    high = np.abs(downscale_restore(image, 112).astype(float) - image).mean()
    low = np.abs(downscale_restore(image, 40).astype(float) - image).mean()
    assert low > high


def test_blur_reduces_variance_and_zero_sigma_is_a_copy():
    image = synthetic_image(size=64)
    np.testing.assert_array_equal(apply_gaussian_blur(image, 0.0), image)
    assert apply_gaussian_blur(image, 2.0).astype(float).var() < image.astype(float).var()


def test_jpeg_round_trip_keeps_shape_and_worse_quality_differs_more():
    image = synthetic_image(size=96)
    good = apply_jpeg(image, 90)
    poor = apply_jpeg(image, 30)
    assert good.shape == image.shape and poor.shape == image.shape
    assert np.abs(poor.astype(float) - image).mean() > np.abs(good.astype(float) - image).mean()


# ---------------------------------------------------------------- occlusion


@pytest.mark.parametrize("kind", ["eyes", "mouth"])
def test_occlusion_patch_area_grows_with_the_requested_fraction(kind):
    areas = [np.prod(np.subtract(occluded_patch(kind, fraction, TEMPLATE_KPS)[2:],
                                 occluded_patch(kind, fraction, TEMPLATE_KPS)[:2]))
             for fraction in (0.05, 0.10, 0.20, 0.30)]
    assert areas == sorted(areas)
    face_area = (FACE_SCALE_INTEROCULAR * 35.2372) ** 2
    assert areas[0] / face_area == pytest.approx(0.05, rel=0.05)
    assert areas[3] / face_area == pytest.approx(0.30, rel=0.05)


def test_eye_patch_is_centred_between_the_eye_landmarks():
    rectangle = occluded_patch("eyes", 0.10, TEMPLATE_KPS)
    centre_x = (rectangle[0] + rectangle[2]) / 2
    assert centre_x == pytest.approx(TEMPLATE_KPS[:2, 0].mean(), abs=0.5)


def test_mouth_patch_sits_below_the_eye_patch():
    eyes = occluded_patch("eyes", 0.10, TEMPLATE_KPS)
    mouth = occluded_patch("mouth", 0.10, TEMPLATE_KPS)
    assert (mouth[1] + mouth[3]) / 2 > (eyes[1] + eyes[3]) / 2


def test_mask_band_covers_more_of_the_lower_face_than_sunglasses():
    sunglasses = band_patch("sunglasses", TEMPLATE_KPS)
    mask = band_patch("mask", TEMPLATE_KPS)
    assert (mask[3] - mask[1]) > (sunglasses[3] - sunglasses[1])
    assert (mask[1] + mask[3]) / 2 > (sunglasses[1] + sunglasses[3]) / 2


def test_occlusion_only_changes_pixels_inside_the_patch():
    crop = face_like_crop()
    occluded = apply_occlusion(crop, (30.0, 44.0, 82.0, 60.0))
    unchanged = occluded.copy()
    unchanged[44:60, 30:82] = crop[44:60, 30:82]
    np.testing.assert_array_equal(unchanged, crop)
    np.testing.assert_array_equal(occluded[44:60, 30:82], OCCLUSION_FILL)


def test_occlusion_is_pushed_inside_the_image_at_the_border():
    crop = face_like_crop()
    occluded = apply_occlusion(crop, (-40.0, -40.0, 10.0, 10.0))
    assert occluded.shape == crop.shape
    assert (occluded[:10, :10] == OCCLUSION_FILL).any()


# ---------------------------------------------------------------- grid


def test_grid_is_frozen_deterministic_and_unique():
    first = perturbation_grid()
    second = perturbation_grid()
    assert first == second
    levels = [entry["level"] for entry in first]
    assert len(levels) == len(set(levels))
    assert levels[0] == "original"
    assert {entry["family"] for entry in first} == {"none", "pose", "illumination",
                                                    "resolution", "occlusion"}


def test_grid_records_every_requested_parameter_value():
    grid = perturbation_grid()
    gammas = {entry["params"]["gamma"] for entry in grid if entry["subfamily"] == "gamma"}
    assert gammas == {0.5, 0.7, 1.0, 1.4, 2.0}
    sizes = {entry["params"]["short_side_px"] for entry in grid
             if entry["subfamily"] == "downscale"}
    assert sizes == {224, 160, 112, 80, 56, 40}
    sigmas = {entry["params"]["sigma"] for entry in grid if entry["subfamily"] == "blur"}
    assert sigmas == {0.5, 1.0, 2.0, 3.0}
    qualities = {entry["params"]["quality"] for entry in grid if entry["subfamily"] == "jpeg"}
    assert qualities == {30, 50, 70, 90}
    fractions = {entry["params"]["area_fraction"] for entry in grid
                 if entry["subfamily"] in ("eyes", "mouth")}
    assert fractions == {0.05, 0.10, 0.20, 0.30}
    angles = {entry["params"]["angle_deg"] for entry in grid if entry["family"] == "pose"
              if entry["params"]["axis"] == "yaw"}
    assert angles == {10, 20, 30, 40, 50, -10, -20, -30, -40, -50}


def test_realistic_tier_excludes_the_harsh_levels_and_keeps_the_neutral_ones():
    grid = {entry["level"]: entry["realistic"] for entry in perturbation_grid()}
    assert grid["original"] is True
    assert grid["gamma_1"] is True
    assert grid["gamma_0.5"] is False
    assert grid["yaw+20"] is True
    assert grid["yaw+30"] is False
    assert grid["roll-20"] is True
    assert grid["jpeg_q50"] is True
    assert grid["jpeg_q30"] is False
    assert grid["blur_sigma_1"] is True
    assert grid["blur_sigma_2"] is False
    assert grid["downscale_160px"] is True
    assert grid["downscale_112px"] is False
    assert grid["eyes_10pct"] is True
    assert grid["eyes_20pct"] is False
    assert grid["sunglasses_band"] is False
    assert grid["mask_band"] is False
    assert grid["shading_0.5_at_0deg"] is True
    assert grid["shading_0.75_at_0deg"] is False


def test_every_family_has_at_least_one_realistic_level():
    shipped = {entry["subfamily"] for entry in perturbation_grid() if entry["realistic"]}
    assert shipped == {"none", "yaw", "pitch", "roll", "gamma", "gain_offset",
                       "temperature", "shading", "downscale", "blur", "jpeg", "eyes", "mouth"}


def test_stable_rng_is_order_independent_and_seed_sensitive():
    first = stable_rng(BASE_SEED, "occlusion", "eyes_10pct").random(3)
    again = stable_rng(BASE_SEED, "occlusion", "eyes_10pct").random(3)
    different = stable_rng(BASE_SEED + 1, "occlusion", "eyes_10pct").random(3)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, different)


def test_every_grid_entry_applies_and_stays_in_uint8():
    image = cv2.resize(face_like_crop(), (256, 256))
    for entry in perturbation_grid():
        perturbed = apply_perturbation(image, entry, kps=None if entry["family"] != "occlusion"
                                       else image.shape and np.array([[70, 100], [170, 100],
                                                                     [120, 150], [85, 190],
                                                                     [155, 190]], float))
        assert perturbed.dtype == np.uint8
        assert perturbed.size > 0


def test_occlusion_without_landmarks_is_rejected():
    with pytest.raises(ValueError):
        apply_perturbation(face_like_crop(), grid_entry("occlusion", "eyes_10pct"))


def test_unknown_perturbation_is_rejected():
    with pytest.raises(ValueError):
        apply_perturbation(face_like_crop(), {"family": "vibes", "subfamily": "x",
                                              "level": "x", "params": {}, "realistic": True})


def test_occlusion_jitter_is_deterministic_but_not_identical_across_levels():
    crop = cv2.resize(face_like_crop(), (224, 224))
    kps = TEMPLATE_KPS * 2
    first = apply_perturbation(crop, grid_entry("occlusion", "eyes_10pct"), kps=kps)
    again = apply_perturbation(crop, grid_entry("occlusion", "eyes_10pct"), kps=kps)
    other = apply_perturbation(crop, grid_entry("occlusion", "mouth_10pct"), kps=kps)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


# ---------------------------------------------------------------- aggregation


def test_aggregate_reports_mean_and_sample_deviation_over_present_values():
    stats = aggregate([0.9, 0.8, 0.7, None])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(0.8)
    assert stats["std"] == pytest.approx(0.1)
    assert aggregate([0.5])["std"] == 0.0
    assert aggregate([None, None]) == {"n": 0, "mean": None, "std": None}


def test_summary_turns_cosines_into_drops_and_counts_detections():
    entry = grid_entry("pose", "yaw+10")
    summary = summarise_level("end_to_end", entry, {
        "person_maya": {"cosine": 1.0, "detected": True, "n_faces": 1},
        "person_priya": {"cosine": 0.6, "detected": True, "n_faces": 1},
        "person_jonah": {"cosine": None, "detected": False, "n_faces": 0},
    })
    assert summary["mean_cosine"] == pytest.approx(0.8)
    assert summary["mean_drop"] == pytest.approx(0.2)
    assert summary["per_identity_drop"]["person_jonah"] is None
    assert summary["identities_detected"] == 2
    assert summary["identities_expected"] == 3
    assert summary["realistic"] is True


def rows_for_floor(end_to_end_pairs, embed_only_pairs=()):
    summaries = [summarise_level("end_to_end", grid_entry("none", "original"),
                                 {"a": {"cosine": 1.0, "detected": True, "n_faces": 1}})]
    for level, value, realistic in end_to_end_pairs:
        entry = grid_entry("illumination", level)
        summary = summarise_level("end_to_end", {**entry, "realistic": realistic},
                                  {"a": {"cosine": value, "detected": True, "n_faces": 1}})
        summaries.append(summary)
    for level, value in embed_only_pairs:
        entry = grid_entry("illumination", level)
        summaries.append(summarise_level("embed_only", entry,
                                         {"a": {"cosine": value, "detected": True,
                                                "n_faces": 1}}))
    return summaries


def test_floor_uses_only_realistic_rows_and_is_one_minus_the_worst_cosine():
    floor = select_floor(rows_for_floor([("gamma_1.4", 0.90, True),
                                         ("gamma_2", 0.10, False)]))
    assert floor["T"] == pytest.approx(0.10)
    assert floor["S_floor"] == pytest.approx(0.90)
    assert floor["T_all"] == pytest.approx(0.90)
    assert floor["S_floor_all"] == pytest.approx(0.10)
    assert floor["driving_level"] == "gamma_1.4"


def test_floor_reports_the_photometric_envelope_separately_from_occlusion():
    def row(family, level, value):
        return summarise_level("end_to_end", grid_entry(family, level),
                               {"a": {"cosine": value, "detected": True, "n_faces": 1}})

    floor = select_floor([row("none", "original", 1.0),
                          row("occlusion", "eyes_10pct", 0.70),
                          row("illumination", "gamma_1.4", 0.90)])
    assert floor["T"] == pytest.approx(0.30)
    assert floor["T_without_occlusion"] == pytest.approx(0.10)
    assert floor["T_p90_without_occlusion"] <= floor["T_without_occlusion"]


def test_floor_ignores_the_embed_only_regime():
    floor = select_floor(rows_for_floor([("gamma_1.4", 0.95, True)],
                                        [("gamma_0.5", 0.05)]))
    assert floor["T"] == pytest.approx(0.05)


def test_floor_refuses_to_run_when_the_control_did_not_reproduce_the_baseline():
    summaries = rows_for_floor([("gamma_1.4", 0.90, True)])
    summaries[0]["per_identity_drop"]["a"] = 0.01
    with pytest.raises(ValueError):
        select_floor(summaries)


def test_floor_requires_the_unperturbed_control_row():
    summaries = rows_for_floor([("gamma_1.4", 0.90, True)])
    with pytest.raises(ValueError):
        select_floor([row for row in summaries if row["level"] != "original"])


def test_between_identity_gap_uses_the_most_similar_impostor():
    gap = between_identity_gap([[1.0, 0.25, 0.10], [0.25, 1.0, 0.05], [0.10, 0.05, 1.0]],
                               ["a", "b", "c"])
    assert gap["max_cross_identity_cosine"] == pytest.approx(0.25)
    assert gap["identity_swap_drop_hardest"] == pytest.approx(0.75)
    assert gap["hardest_pair"] == ["a", "b"]
    assert gap["min_cross_identity_cosine"] == pytest.approx(0.05)


def test_signal_to_noise_expresses_nuisance_as_a_share_of_an_identity_swap():
    gap = {"identity_swap_drop_hardest": 0.75}
    ratio = signal_to_noise({"T": 0.25}, gap)
    assert ratio["ratio"] == pytest.approx(3.0)
    assert ratio["nuisance_share"] == pytest.approx(1 / 3)
    assert signal_to_noise({"T": 0.0}, gap)["ratio"] is None


def test_family_curve_filters_regime_and_subfamily():
    summaries = rows_for_floor([("gamma_1.4", 0.9, True)], [("gamma_0.5", 0.4)])
    assert len(family_curve(summaries, "end_to_end", "illumination")) == 1
    assert len(family_curve(summaries, "end_to_end", "illumination", "gamma")) == 1
    assert len(family_curve(summaries, "end_to_end", "pose")) == 0
    assert len(family_curve(summaries, "embed_only", "illumination")) == 1


def test_realistic_family_maxima_reports_the_worst_level_per_subfamily():
    maxima = realistic_family_maxima(rows_for_floor([("gamma_1.4", 0.90, True),
                                                     ("gamma_0.7", 0.95, True),
                                                     ("gamma_2", 0.10, False)]))
    assert maxima["illumination/gamma"]["max_drop"] == pytest.approx(0.10)
    assert maxima["illumination/gamma"]["level"] == "gamma_1.4"


# ---------------------------------------------------------------- assignment verdicts


def scores_fixture():
    return {
        "matrix_row_order": ["person_maya", "person_priya", "person_jonah"],
        "scenes": [{"photo_index": 0, "month": 1, "passes": [{
            "pass_index": 0,
            "cosine_matrix": [[0.30, 0.02], [0.03, 0.04], [0.05, 0.12]],
            "candidates": [
                {"person_id": "person_maya", "face_index": 0, "similarity": 0.30},
                {"person_id": "person_priya", "face_index": 1, "similarity": 0.04},
                {"person_id": "person_jonah", "face_index": None, "similarity": None},
            ]}]}],
    }


def test_assignment_margin_is_measured_against_the_best_other_identity():
    classification = classify_assignments(scores_fixture(), {"S_floor": 0.2, "T": 0.8})
    maya = classification["rows"][0]
    assert maya["best_other_identity_cosine"] == pytest.approx(0.05)
    assert maya["margin"] == pytest.approx(0.25)
    assert maya["drop_from_perfect"] == pytest.approx(0.70)
    # The assigned identity need not be the best-scoring one for that face.
    priya = classification["rows"][1]
    assert priya["best_other_identity_cosine"] == pytest.approx(0.12)
    assert priya["margin"] == pytest.approx(-0.08)


def test_negative_margin_is_kept_and_counted_as_not_rank_one():
    classification = classify_assignments(scores_fixture(), {"S_floor": 0.0, "T": 1.0})
    assert classification["rows"][1]["margin"] < 0
    assert classification["rank1"] == 1
    assert classification["rows"][0]["margin"] > 0


def test_assignments_below_the_floor_are_marked_beyond_the_envelope():
    classification = classify_assignments(scores_fixture(), {"S_floor": 0.2, "T": 0.8})
    statuses = {row["person_id"]: row["status"] for row in classification["rows"]}
    assert statuses["person_maya"] == "inside_envelope"
    assert statuses["person_priya"] == "beyond_envelope"
    assert classification["scored"] == 2
    assert classification["not_detected"] == 1
    assert classification["inside_envelope"] == 1
    assert classification["beyond_envelope"] == 1
    assert classification["rank1"] == 1
    assert classification["rank1_margin_below_T"] == 1


def test_missing_score_is_reported_as_not_detected_not_as_a_failure():
    classification = classify_assignments(scores_fixture(), {"S_floor": 0.2, "T": 0.8})
    missing = classification["rows"][2]
    assert missing["status"] == "not_detected"
    assert missing["similarity"] is None


# ---------------------------------------------------------------- iteration 01


FACE_SCORES_FIXTURE = """condition,month,pass,person,similarity,face_index,image
historical,1,1,Maya Chen,0.4456,0,p1.png
historical,1,2,Maya Chen,0.1453,0,p1b.png
historical,2,1,Maya Chen,0.3081,0,p2.png
historical,2,2,Maya Chen,0.0618,0,p2b.png
historical,5,1,Maya Chen,0.3098,0,p5.png
historical,8,1,Maya Chen,0.3418,0,p8.png
historical,8,2,Maya Chen,0.2231,0,p8b.png
historical,10,1,Maya Chen,0.1091,0,p10.png
historical,10,2,Maya Chen,0.0575,0,p10b.png
historical,1,1,Jonah,0.1519,1,p1.png
"""


def test_face_score_csv_parses_blank_similarity_as_no_measurement():
    rows = parse_face_scores(FACE_SCORES_FIXTURE.replace("0.4456", ""))
    assert rows[0]["similarity"] is None
    assert rows[0]["person"] == "Maya Chen"
    assert rows[1]["similarity"] == pytest.approx(0.1453)


def test_multipass_drops_skip_single_pass_photos_and_other_people():
    rows = parse_face_scores(FACE_SCORES_FIXTURE)
    drops = multipass_drops(rows, "historical")
    assert [record["month"] for record in drops] == [1, 2, 8, 10]
    assert drops[0]["drop"] == pytest.approx(0.4456 - 0.1453)
    assert drops[-1]["drop"] == pytest.approx(0.1091 - 0.0575)


def test_iteration01_verdicts_flag_drops_above_the_floor_and_below_double_it():
    rows = parse_face_scores(FACE_SCORES_FIXTURE)
    check = check_iteration01(rows, {"T": 0.10, "S_floor": 0.90})
    historical = check["conditions"]["historical"]
    assert historical["n_photos"] == 4
    assert historical["all_fell"] is True
    assert historical["n_exceeding_T"] == 3
    assert historical["n_exceeding_2T"] == 2
    assert historical["photos"][-1]["exceeds_T"] is False
    assert check["two_T"] == pytest.approx(0.20)


def test_iteration01_reports_a_rise_as_not_falling():
    rows = parse_face_scores(FACE_SCORES_FIXTURE.replace(
        "historical,10,2,Maya Chen,0.0575", "historical,10,2,Maya Chen,0.1284"))
    check = check_iteration01(rows, {"T": 0.10, "S_floor": 0.90})
    historical = check["conditions"]["historical"]
    assert historical["all_fell"] is False
    assert historical["photos"][-1]["drop"] < 0


def test_iteration01_absent_condition_reports_no_photos():
    check = check_iteration01(parse_face_scores(FACE_SCORES_FIXTURE), {"T": 0.1, "S_floor": 0.9})
    assert check["conditions"]["control"]["n_photos"] == 0
    assert check["conditions"]["control"]["mean_drop"] is None
    assert check["claim"].startswith("In the historical pilot")


# ---------------------------------------------------------------- CLI and I/O


def test_cli_defaults_point_at_the_frozen_inputs():
    args = parse_args([])
    assert args.project_root == PROJECT
    assert args.images == PROJECT / "output/images"
    assert args.scores == HERE / "results/scores.json"
    assert args.output == HERE / "results_invariance"
    assert args.seed == BASE_SEED


def test_cli_rejects_a_changed_seed_so_results_stay_comparable():
    with pytest.raises(SystemExit) as error:
        parse_args(["--seed", str(BASE_SEED + 1)])
    assert error.value.code == 2


def test_cli_rejects_images_outside_the_hash_root(tmp_path):
    with pytest.raises(SystemExit) as error:
        parse_args(["--project-root", str(tmp_path / "experiment"),
                    "--images", str(tmp_path / "outside")])
    assert error.value.code == 2


def test_sha256_reads_the_file_bytes_as_hex(tmp_path):
    path = tmp_path / "payload.txt"
    path.write_text("invariance\n")
    expected = hashlib.sha256(b"invariance\n").hexdigest()
    assert sha256(path) == expected
    assert len(sha256(path)) == 64


def test_project_paths_are_inside_the_repository():
    assert HERE == PROJECT / "evaluation"
    assert HERE.is_relative_to(PROJECT)


# ---------------------------------------------------------------- model-backed


@needs_weights
def test_harness_reproduces_the_frozen_reference_similarities():
    from face_invariance import detect_and_embed, load_analysis

    provenance = json.loads((HERE / "model-provenance.json").read_text())
    app = load_analysis(MODELS, provenance)
    scores = json.loads((HERE / "results/scores.json").read_text())
    assets = {"person_maya": "000_person_maya.png", "person_priya": "001_person_priya.png",
              "person_jonah": "002_person_jonah.png"}
    embeddings = []
    for identity in scores["matrix_row_order"]:
        image = cv2.imread(str(PROJECT / "output/images/references/people" / assets[identity]))
        embeddings.append(detect_and_embed(image, app)["cosine_input"])
    matrix = np.array(embeddings) @ np.array(embeddings).T
    assert np.abs(matrix - np.array(scores["reference_cosine_matrix"])).max() < 1e-6


@needs_weights
def test_embed_only_baseline_equals_the_end_to_end_baseline():
    from face_invariance import detect_and_embed, load_analysis

    provenance = json.loads((HERE / "model-provenance.json").read_text())
    app = load_analysis(MODELS, provenance)
    image = cv2.imread(str(PROJECT / "output/images/references/people/000_person_maya.png"))
    record = detect_and_embed(image, app)
    from_crop = embed_crop(record["aligned"], app.models["recognition"])
    np.testing.assert_allclose(from_crop, record["cosine_input"], atol=1e-7)


@needs_weights
def test_identity_perturbation_is_a_no_op_for_both_regimes():
    from face_invariance import detect_and_embed, load_analysis

    provenance = json.loads((HERE / "model-provenance.json").read_text())
    app = load_analysis(MODELS, provenance)
    image = cv2.imread(str(PROJECT / "output/images/references/people/002_person_jonah.png"))
    record = detect_and_embed(image, app)
    entry = grid_entry("none", "original")
    perturbed = apply_perturbation(image, entry, kps=record["kps"])
    assert cosine(record["cosine_input"], detect_and_embed(perturbed, app)["cosine_input"]) == \
        pytest.approx(1.0, abs=1e-12)
    crop = apply_perturbation(record["aligned"], entry, kps=record["kps"])
    assert cosine(record["cosine_input"], embed_crop(crop, app.models["recognition"])) == \
        pytest.approx(1.0, abs=1e-12)
