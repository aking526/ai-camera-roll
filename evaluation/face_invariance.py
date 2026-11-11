"""Measure how far ArcFace cosine similarity moves when identity does not change.

The pipeline scored by ``score_faces.py`` reports a raw cosine similarity between
a canonical portrait and a detected scene face. That number has no calibrated
identity threshold and no measured noise floor, so a drop between two runs cannot
currently be attributed to identity change. This script measures the floor
empirically: it applies deterministic, identity-preserving nuisance to the
canonical portraits and records how far the cosine moves.

Two regimes are measured separately.

* ``end_to_end``: perturb the reference image, then re-run detect -> norm_crop ->
  embed. Includes detector and alignment sensitivity.
* ``embed_only``: perturb the already-aligned 112x112 crop, then embed. Isolates
  the embedder.

The 2D poses here are a planar perspective warp of the whole image, not a 3D
render, so they cannot reproduce self-occlusion; pose sensitivity is therefore
optimistic. Cosine similarity is not a probability and this measures metric
sensitivity, not human-perceived identity.

Nothing here trains or fine-tunes a model and no weights are downloaded.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# Frozen before any run: recorded in the results JSON so the grid cannot drift.
BASE_SEED = 20261104
# Focal length as a multiple of image width, i.e. a camera property: 0.7 gives a
# 71 degree horizontal field, an ordinary phone photo. The aligned-crop regime
# inherits the focal-to-face-width ratio the face had in its photo, so a rotation
# distorts the face identically in both regimes instead of twice as hard in the
# crop just because the crop is zoomed in.
FOCAL_PER_FRAME_WIDTH = 0.7
ALIGNED_CROP_SIZE = 112
# Where norm_crop places the five landmarks inside the 112x112 aligned crop.
ALIGNED_TEMPLATE_KPS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                                 [41.5493, 92.3655], [70.7299, 92.2041]])
POSE_ANGLES = (10, 20, 30, 40, 50)
POSE_AXES = ("yaw", "pitch", "roll")
GAMMAS = (0.5, 0.7, 1.0, 1.4, 2.0)
GAIN_OFFSETS = ((0.8, 20.0), (0.8, -20.0), (1.25, 20.0), (1.25, -20.0), (0.6, 0.0), (1.6, 0.0))
TEMPERATURES = (3000.0, 4500.0, 9000.0)
SHADING = ((0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (0.5, 90.0))
RESOLUTIONS = (224, 160, 112, 80, 56, 40)
BLURS = (0.5, 1.0, 2.0, 3.0)
JPEG_QUALITIES = (30, 50, 70, 90)
OCCLUSION_AREAS = (0.05, 0.10, 0.20, 0.30)
OCCLUSION_KINDS = ("eyes", "mouth")

# A face box is taken as this multiple of the interocular distance. For the
# 112x112 ArcFace template the interocular distance is 35.24 px, so 3.2 lands on
# the crop edge and occlusion percentages stay comparable across regimes.
FACE_SCALE_INTEROCULAR = 3.2
OCCLUSION_FILL = 128

# "Realistic" tier, frozen before the run. A perturbation is called realistic
# when a human would still name the same person, because it matches an ordinary
# photographic variation: casual head turn, ordinary indoor/outdoor light,
# preview-sized rendering, soft focus, social-media recompression, or a small
# patch over one eye or the mouth. Everything outside the tier is still measured
# and reported, but is excluded from the noise floor T. The exclusion of eye and
# mouth occlusion above 10% and of the sunglass/mask bands is a judgement call:
# those remove the cues the embedder relies on, so including them would inflate T
# past the point where any real score change could be detected.
REALISTIC_POSE_MAX = 20
REALISTIC_GAMMA = (0.7, 1.4)
REALISTIC_GAIN = (0.8, 1.25)
REALISTIC_TEMPERATURE = TEMPERATURES
REALISTIC_SHADING_MAX = 0.5
REALISTIC_RESOLUTION_MIN = 160
REALISTIC_BLUR_MAX = 1.0
REALISTIC_JPEG_MIN = 50
REALISTIC_OCCLUSION_MAX = 0.10

IDENTITY_PREFIX = "person_"
ITERATION_01_CLAIM = (
    "In the historical pilot, Maya's assigned ArcFace similarity fell after the "
    "second pass in all four multi-pass scenes."
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_rng(base_seed, *parts):
    """Generator seeded from strings, so variants do not depend on call order."""
    key = "|".join([str(base_seed), *[str(part) for part in parts]]).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


# --------------------------------------------------------------------------
# Perturbation primitives (no model, no I/O; unit-testable)
# --------------------------------------------------------------------------


def as_uint8(image):
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def cosine(first, second):
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))


def _rotation(yaw_deg, pitch_deg, roll_deg):
    yaw, pitch, roll = (math.radians(value) for value in (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return r_roll @ r_yaw @ r_pitch


def project_corners(width, height, yaw_deg, pitch_deg, roll_deg, focal_px, centre=None):
    """Re-project the four image corners after rotating the portrait plane.

    The rotation axis passes through ``centre``, by default the image centre and
    for a pose perturbation the head, which stands in for a head turn rather than
    a camera pan or tilt. This is a planar approximation of a 3D pose change: no
    self-occlusion, and the turned surface keeps its original shading because no
    geometry is actually re-lit.
    """
    focal = float(focal_px)
    centre_x, centre_y = centre if centre is not None else ((width - 1) / 2.0,
                                                           (height - 1) / 2.0)
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], float)
    points = np.column_stack([corners[:, 0] - centre_x, corners[:, 1] - centre_y,
                              np.zeros(4)])
    rotated = points @ _rotation(yaw_deg, pitch_deg, roll_deg).T
    depth = focal + rotated[:, 2]
    if np.any(depth <= 1e-6):
        raise ValueError("Pose puts part of the image plane behind the camera.")
    projected = np.column_stack([focal * rotated[:, 0] / depth + centre_x,
                                 focal * rotated[:, 1] / depth + centre_y])
    return corners, projected


def perspective_warp(image, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, focal_px=None,
                     centre=None):
    """Rotate the portrait plane. Returns (warped image, projected corner quad)."""
    height, width = image.shape[:2]
    if focal_px is None:
        focal_px = FOCAL_PER_FRAME_WIDTH * width
    corners, projected = project_corners(width, height, yaw_deg, pitch_deg, roll_deg,
                                         focal_px, centre)
    if np.allclose(projected, corners):
        return image.copy(), projected
    low = projected.min(axis=0)
    high = projected.max(axis=0)
    size = (int(math.ceil(high[0] - low[0])) + 1, int(math.ceil(high[1] - low[1])) + 1)
    matrix = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]], float) @ cv2.getPerspectiveTransform(
        corners.astype(np.float32), projected.astype(np.float32))
    warped = cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, projected


def apply_gamma(image, gamma):
    """Tone-curve change: gamma < 1 brightens midtones, gamma > 1 darkens them."""
    table = np.clip(np.power(np.arange(256, dtype=float) / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, table.astype(np.uint8))


def apply_gain_offset(image, gain, offset):
    """Brightness (offset) and contrast (gain) about mid-grey."""
    source = image.astype(np.float64)
    return as_uint8(gain * (source - 128.0) + 128.0 + offset)


def color_temperature_gains(kelvin):
    """Per-channel multipliers for an approximate blackbody illuminant.

    Tanner Helland's fit, normalised so the green channel is unchanged. Values
    above 1 brighten, so a warm 3000 K tint boosts red and cuts blue.
    """
    scaled = float(np.clip(kelvin, 1000.0, 40000.0)) / 100.0
    if scaled <= 66:
        red = 255.0
    else:
        red = 329.698727446 * (scaled - 60) ** -0.1332047592
    if scaled <= 66:
        green = 99.4708025861 * math.log(scaled) - 161.1195681661
    else:
        green = 288.1221695283 * (scaled - 60) ** -0.0755148492
    if scaled >= 66:
        blue = 255.0
    elif scaled <= 19:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(scaled - 10) - 305.0447927307
    red, green, blue = (float(np.clip(value, 0.0, 255.0)) for value in (red, green, blue))
    if green <= 1e-6:
        raise ValueError("Colour temperature outside the fitted range.")
    return red / green, 1.0, blue / green


def apply_color_temperature(image, kelvin):
    gains = np.array(color_temperature_gains(kelvin))
    return as_uint8(image.astype(np.float64) * gains.reshape(1, 1, 3))


def apply_lighting_gradient(image, strength, angle_deg):
    """Directional shading ramp, a stand-in for a single off-camera light."""
    height, width = image.shape[:2]
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    x = cols / max(width - 1, 1) - 0.5
    y = rows / max(height - 1, 1) - 0.5
    radians = math.radians(angle_deg)
    ramp = x * math.cos(radians) + y * math.sin(radians)
    peak = max(np.abs(ramp).max(), 1e-6)
    factor = 1.0 + strength * ramp / peak
    return as_uint8(image.astype(np.float64) * factor[..., None])


def downscale_restore(image, size):
    """Render at `size` pixels on the short side, then restore the original size."""
    height, width = image.shape[:2]
    target = int(min(size, min(height, width)))
    scale = target / min(height, width)
    small_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_gaussian_blur(image, sigma):
    if sigma <= 0:
        return image.copy()
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)


def apply_jpeg(image, quality):
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise OSError("JPEG encode failed.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise OSError("JPEG decode failed.")
    return decoded


def occluded_patch(kind, area_fraction, kps):
    """Rectangle covering `area_fraction` of the face over an eye or mouth region.

    ``kps`` is the five-point landmark set (left eye, right eye, nose, left mouth,
    right mouth). Area is measured against a face box whose side is
    ``FACE_SCALE_INTEROCULAR`` interocular distances, which is the 112x112
    aligned-crop box, so percentages mean the same thing in both regimes.
    """
    kps = np.asarray(kps, dtype=float)
    interocular = float(np.linalg.norm(kps[1] - kps[0]))
    face_area = (FACE_SCALE_INTEROCULAR * interocular) ** 2
    area = area_fraction * face_area
    aspect = 1.8 if kind == "eyes" else 1.4
    width = math.sqrt(area * aspect)
    height = width / aspect
    centre = kps[:2].mean(axis=0) if kind == "eyes" else kps[3:5].mean(axis=0)
    return (centre[0] - width / 2.0, centre[1] - height / 2.0,
            centre[0] + width / 2.0, centre[1] + height / 2.0)


def band_patch(kind, kps):
    """Fixed-shape sunglasses or surgical-mask band."""
    kps = np.asarray(kps, dtype=float)
    interocular = float(np.linalg.norm(kps[1] - kps[0]))
    if kind == "sunglasses":
        centre = kps[:2].mean(axis=0)
        width, height = 2.6 * interocular, 0.35 * interocular
    elif kind == "mask":
        centre = (kps[2] + kps[3:5].mean(axis=0)) / 2.0
        width, height = 2.2 * interocular, 1.2 * interocular
    else:
        raise ValueError(f"Unknown band: {kind}")
    return (centre[0] - width / 2.0, centre[1] - height / 2.0,
            centre[0] + width / 2.0, centre[1] + height / 2.0)


def apply_occlusion(image, rectangle, fill=OCCLUSION_FILL):
    height, width = image.shape[:2]
    x0 = int(np.clip(math.floor(rectangle[0]), 0, width - 1))
    y0 = int(np.clip(math.floor(rectangle[1]), 0, height - 1))
    x1 = int(np.clip(math.ceil(rectangle[2]), x0 + 1, width))
    y1 = int(np.clip(math.ceil(rectangle[3]), y0 + 1, height))
    occluded = image.copy()
    occluded[y0:y1, x0:x1] = fill
    return occluded


# --------------------------------------------------------------------------
# Frozen grid
# --------------------------------------------------------------------------


def perturbation_grid():
    """The frozen perturbation list. Order is part of the record."""
    grid = [{"family": "none", "subfamily": "none", "level": "original",
             "params": {}, "realistic": True}]
    for angle in POSE_ANGLES:
        for axis in POSE_AXES:
            for sign in (1, -1):
                grid.append({"family": "pose", "subfamily": axis,
                             "level": f"{axis}{sign * angle:+d}",
                             "params": {"axis": axis, "angle_deg": sign * angle},
                             "realistic": angle <= REALISTIC_POSE_MAX})
    for gamma in GAMMAS:
        grid.append({"family": "illumination", "subfamily": "gamma",
                     "level": f"gamma_{gamma:g}", "params": {"gamma": gamma},
                     "realistic": REALISTIC_GAMMA[0] <= gamma <= REALISTIC_GAMMA[1]})
    for gain, offset in GAIN_OFFSETS:
        grid.append({"family": "illumination", "subfamily": "gain_offset",
                     "level": f"gain_{gain:g}_offset_{offset:+g}",
                     "params": {"gain": gain, "offset": offset},
                     "realistic": REALISTIC_GAIN[0] <= gain <= REALISTIC_GAIN[1]})
    for kelvin in TEMPERATURES:
        grid.append({"family": "illumination", "subfamily": "temperature",
                     "level": f"temperature_{kelvin:g}K", "params": {"kelvin": kelvin},
                     "realistic": kelvin in REALISTIC_TEMPERATURE})
    for strength, angle in SHADING:
        grid.append({"family": "illumination", "subfamily": "shading",
                     "level": f"shading_{strength:g}_at_{angle:g}deg",
                     "params": {"strength": strength, "angle_deg": angle},
                     "realistic": strength <= REALISTIC_SHADING_MAX})
    for size in RESOLUTIONS:
        grid.append({"family": "resolution", "subfamily": "downscale",
                     "level": f"downscale_{size}px", "params": {"short_side_px": size},
                     "realistic": size >= REALISTIC_RESOLUTION_MIN})
    for sigma in BLURS:
        grid.append({"family": "resolution", "subfamily": "blur",
                     "level": f"blur_sigma_{sigma:g}", "params": {"sigma": sigma},
                     "realistic": sigma <= REALISTIC_BLUR_MAX})
    for quality in JPEG_QUALITIES:
        grid.append({"family": "resolution", "subfamily": "jpeg",
                     "level": f"jpeg_q{quality}", "params": {"quality": quality},
                     "realistic": quality >= REALISTIC_JPEG_MIN})
    for kind in OCCLUSION_KINDS:
        for fraction in OCCLUSION_AREAS:
            grid.append({"family": "occlusion", "subfamily": kind,
                         "level": f"{kind}_{int(round(fraction * 100))}pct",
                         "params": {"kind": kind, "area_fraction": fraction},
                         "realistic": fraction <= REALISTIC_OCCLUSION_MAX + 1e-12})
    for kind in ("sunglasses", "mask"):
        grid.append({"family": "occlusion", "subfamily": kind, "level": f"{kind}_band",
                     "params": {"kind": kind}, "realistic": False})
    levels = [entry["level"] for entry in grid]
    if len(set(levels)) != len(levels):
        raise ValueError("Perturbation levels must be unique.")
    return grid


def apply_perturbation(image, entry, kps=None, focal_px=None, centre=None, seed=BASE_SEED):
    """Apply one frozen grid entry.

    ``kps`` is required for occlusion entries and must be in the image's own pixel
    frame. ``focal_px`` and ``centre`` set the pose distortion: the focal length in
    pixels and the point the portrait turns about.
    """
    family, subfamily, params = entry["family"], entry["subfamily"], entry["params"]
    if family == "none":
        return image.copy()
    if family == "pose":
        warped, _ = perspective_warp(image, focal_px=focal_px, centre=centre,
                                     **{f"{params['axis']}_deg": params["angle_deg"]})
        return warped
    if family == "illumination":
        if subfamily == "gamma":
            return apply_gamma(image, params["gamma"])
        if subfamily == "gain_offset":
            return apply_gain_offset(image, params["gain"], params["offset"])
        if subfamily == "temperature":
            return apply_color_temperature(image, params["kelvin"])
        if subfamily == "shading":
            return apply_lighting_gradient(image, params["strength"], params["angle_deg"])
    if family == "resolution":
        if subfamily == "downscale":
            return downscale_restore(image, params["short_side_px"])
        if subfamily == "blur":
            return apply_gaussian_blur(image, params["sigma"])
        if subfamily == "jpeg":
            return apply_jpeg(image, params["quality"])
    if family == "occlusion":
        if kps is None:
            raise ValueError("Occlusion perturbations need five-point landmarks.")
        if subfamily in OCCLUSION_KINDS:
            # Seeded jitter keeps the patch over the intended region without
            # making every variant land on exactly the same pixels.
            rng = stable_rng(seed, entry["family"], entry["level"])
            rectangle = occluded_patch(subfamily, params["area_fraction"], kps)
            width = rectangle[2] - rectangle[0]
            height = rectangle[3] - rectangle[1]
            dx = rng.uniform(-0.08, 0.08) * width
            dy = rng.uniform(-0.08, 0.08) * height
            rectangle = (rectangle[0] + dx, rectangle[1] + dy,
                         rectangle[2] + dx, rectangle[3] + dy)
            return apply_occlusion(image, rectangle)
        return apply_occlusion(image, band_patch(subfamily, kps))
    raise ValueError(f"Unknown perturbation: {family}/{subfamily}")


# --------------------------------------------------------------------------
# Aggregation, floor selection, and the two audit tables (all model-free)
# --------------------------------------------------------------------------


def aggregate(values):
    """Mean and sample standard deviation, ignoring identities without a face."""
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"n": 0, "mean": None, "std": None}
    std = float(np.std(present, ddof=1)) if len(present) > 1 else 0.0
    return {"n": len(present), "mean": float(np.mean(present)), "std": std}


def summarise_level(regime, entry, per_identity):
    """One row of the per-family curve: cosines, drops, and detection outcome."""
    cosines = {identity: record["cosine"] for identity, record in per_identity.items()}
    drops = {identity: None if value is None else 1.0 - value
             for identity, value in cosines.items()}
    cosine_stats = aggregate(cosines.values())
    drop_stats = aggregate(drops.values())
    detected = sum(1 for record in per_identity.values() if record["detected"])
    return {
        "regime": regime, "family": entry["family"], "subfamily": entry["subfamily"],
        "level": entry["level"], "params": entry["params"],
        "realistic": entry["realistic"],
        "per_identity_cosine": cosines, "per_identity_drop": drops,
        "mean_cosine": cosine_stats["mean"], "std_cosine": cosine_stats["std"],
        "mean_drop": drop_stats["mean"], "std_drop": drop_stats["std"],
        "identities_detected": detected, "identities_scored": cosine_stats["n"],
        "identities_expected": len(per_identity),
    }


def identity_drops(summary):
    return [value for value in summary["per_identity_drop"].values() if value is not None]


def select_floor(summaries, reference_self_cosine=1.0):
    """Derive T, the identity-equivalent nuisance floor, from measured rows.

    ``T`` is the largest cosine drop any single identity showed under a
    *realistic* perturbation of the same identity, in the end-to-end regime. A
    score can therefore land as low as ``S_floor = 1 - T`` while identity is
    provably unchanged. ``T_all`` uses every perturbation, including the harsh
    ones outside the realistic tier, and is reported so the exclusion is visible.
    """
    end_to_end = [row for row in summaries if row["regime"] == "end_to_end"]
    if not any(row["level"] == "original" for row in end_to_end):
        raise ValueError("The unperturbed control row is required to trust the floor.")
    control = [value for row in end_to_end if row["level"] == "original"
               for value in identity_drops(row)]
    if not control or max(abs(value) for value in control) > 1e-6:
        raise ValueError("Unperturbed control did not reproduce the baseline embedding.")

    realistic = [row for row in end_to_end if row["realistic"]]
    all_rows = [row for row in end_to_end if row["level"] != "original"]
    realistic_drops = [value for row in realistic for value in identity_drops(row)]
    all_drops = [value for row in all_rows for value in identity_drops(row)]
    if not realistic_drops:
        raise ValueError("No realistic perturbation had a detected face.")
    mild_max = float(max(realistic_drops))
    no_occlusion = [value for row in realistic if row["family"] != "occlusion"
                    for value in identity_drops(row)]
    return {
        "T": mild_max,
        "S_floor": reference_self_cosine - mild_max,
        "T_all": float(max(all_drops)),
        "S_floor_all": reference_self_cosine - float(max(all_drops)),
        "T_mean_over_identities": float(max(row["mean_drop"] for row in realistic)),
        "T_without_occlusion": float(max(no_occlusion)) if no_occlusion else None,
        "T_p90": float(np.percentile(realistic_drops, 90)),
        "T_p90_without_occlusion": float(np.percentile(no_occlusion, 90)) if no_occlusion else None,
        "T_median": float(np.median(realistic_drops)),
        "driving_level": max(
            (row for row in realistic if row["per_identity_drop"]),
            key=lambda row: max(identity_drops(row)))["level"],
        "realistic_cells": len(realistic_drops),
        "all_cells": len(all_drops),
        "reference_self_cosine": reference_self_cosine,
    }


def between_identity_gap(reference_cosine_matrix, row_order):
    """Largest similarity between two different identities; the hardest impostor."""
    matrix = np.asarray(reference_cosine_matrix, dtype=float)
    pairs = [(row_order[i], row_order[j], float(matrix[i, j]))
             for i in range(len(row_order)) for j in range(i + 1, len(row_order))]
    hardest = max(pairs, key=lambda pair: pair[2])
    easiest = min(pairs, key=lambda pair: pair[2])
    return {
        "pairs": [{"a": a, "b": b, "cosine": value} for a, b, value in pairs],
        "hardest_pair": [hardest[0], hardest[1]],
        "max_cross_identity_cosine": hardest[2],
        "min_cross_identity_cosine": easiest[2],
        "identity_swap_drop_hardest": 1.0 - hardest[2],
    }


def signal_to_noise(floor, gap):
    """Nuisance drop as a fraction of the drop produced by the hardest impostor."""
    if floor["T"] <= 0:
        return {"ratio": None, "nuisance_share": None}
    return {
        "ratio": gap["identity_swap_drop_hardest"] / floor["T"],
        "nuisance_share": floor["T"] / gap["identity_swap_drop_hardest"],
    }


def classify_assignments(scores, floor):
    """Score-level check of every saved assignment against the same-identity floor.

    A cosine of 1.0 is the only identity-equal anchor available without a
    calibrated threshold, so each score is compared directly with ``S_floor``:
    the lowest cosine observed between two renderings of one identity under
    realistic nuisance. Below that floor the reference and the face are further
    apart than any same-identity pair the grid produced.
    """
    rows = []
    matrix_rows = scores["matrix_row_order"]
    for scene in scores["scenes"]:
        for passing in scene["passes"]:
            matrix = np.asarray(passing["cosine_matrix"], dtype=float)
            for candidate in passing["candidates"]:
                similarity = candidate["similarity"]
                face_index = candidate["face_index"]
                row = {
                    "photo_index": scene["photo_index"], "month": scene["month"],
                    "pass_index": passing["pass_index"],
                    "person_id": candidate["person_id"],
                    "face_index": face_index, "similarity": similarity,
                }
                if similarity is None or face_index is None:
                    row.update({"status": "not_detected", "margin": None,
                                "best_other_identity_cosine": None,
                                "drop_from_perfect": None})
                    rows.append(row)
                    continue
                row_index = matrix_rows.index(candidate["person_id"])
                others = [float(matrix[index, face_index])
                          for index in range(len(matrix_rows)) if index != row_index]
                best_other = max(others)
                row.update({
                    "margin": float(similarity) - best_other,
                    "best_other_identity_cosine": best_other,
                    "drop_from_perfect": 1.0 - float(similarity),
                    "status": ("inside_envelope" if similarity >= floor["S_floor"]
                               else "beyond_envelope"),
                })
                rows.append(row)
    scored = [row for row in rows if row["similarity"] is not None]
    return {
        "rows": rows,
        "scored": len(scored),
        "not_detected": len(rows) - len(scored),
        "inside_envelope": sum(1 for row in scored if row["status"] == "inside_envelope"),
        "beyond_envelope": sum(1 for row in scored if row["status"] == "beyond_envelope"),
        "rank1": sum(1 for row in scored if row["margin"] > 0),
        "rank1_margin_below_T": sum(1 for row in scored if 0 < row["margin"] < floor["T"]),
        "rank1_below_half_T": sum(1 for row in scored
                                  if 0 < row["margin"] < floor["T"] / 2.0),
    }


def parse_face_scores(text):
    rows = []
    for record in csv.DictReader(io.StringIO(text)):
        similarity = (record.get("similarity") or "").strip()
        face_index = (record.get("face_index") or "").strip()
        rows.append({
            "condition": record["condition"], "month": int(record["month"]),
            "pass": int(record["pass"]), "person": record["person"],
            "similarity": float(similarity) if similarity else None,
            "face_index": int(face_index) if face_index else None,
            "image": record["image"],
        })
    return rows


def multipass_drops(rows, condition, person="Maya Chen"):
    """First-pass to second-pass similarity change for one condition."""
    by_month = {}
    for row in rows:
        if row["condition"] != condition or row["person"] != person:
            continue
        by_month.setdefault(row["month"], {})[row["pass"]] = row
    drops = []
    for month in sorted(by_month):
        passes = by_month[month]
        if len(passes) < 2 or 1 not in passes or 2 not in passes:
            continue
        first, second = passes[1], passes[2]
        if first["similarity"] is None or second["similarity"] is None:
            continue
        drops.append({
            "month": month, "pass1": first["similarity"], "pass2": second["similarity"],
            "drop": first["similarity"] - second["similarity"],
            "pass1_face_index": first["face_index"], "pass2_face_index": second["face_index"],
            "pass1_image": first["image"], "pass2_image": second["image"],
        })
    return drops


def check_iteration01(rows, floor):
    """Test the ANALYSIS.md claim that Maya fell in all four multi-pass photos."""
    threshold = floor["T"]
    double = 2.0 * threshold
    verdicts = {}
    for condition in ("historical", "control", "portrait_reference"):
        drops = multipass_drops(rows, condition)
        for record in drops:
            record["exceeds_T"] = record["drop"] > threshold
            record["exceeds_2T"] = record["drop"] > double
            record["pass2_inside_envelope"] = record["pass2"] >= floor["S_floor"]
        all_fell = bool(drops) and all(record["drop"] > 0 for record in drops)
        verdicts[condition] = {
            "photos": drops,
            "n_photos": len(drops),
            "all_fell": all_fell,
            "n_exceeding_T": sum(1 for record in drops if record["exceeds_T"]),
            "n_exceeding_2T": sum(1 for record in drops if record["exceeds_2T"]),
            "mean_drop": float(np.mean([record["drop"] for record in drops])) if drops else None,
        }
    return {"claim": ITERATION_01_CLAIM, "T": threshold, "two_T": double,
            "T_p90": floor.get("T_p90"), "S_floor": floor["S_floor"],
            "conditions": verdicts}


def family_curve(summaries, regime, family, subfamily=None):
    return [row for row in summaries
            if row["regime"] == regime and row["family"] == family
            and (subfamily is None or row["subfamily"] == subfamily)]


def realistic_family_maxima(summaries, regime="end_to_end"):
    """Worst realistic cell per family/subfamily, for the per-family curve report."""
    maxima = {}
    keys = sorted({(row["family"], row["subfamily"]) for row in summaries
                   if row["regime"] == regime})
    for family, subfamily in keys:
        candidates = [row for row in family_curve(summaries, regime, family, subfamily)
                      if row["realistic"] and row["level"] != "original"
                      and identity_drops(row)]
        if not candidates:
            continue
        worst = max(candidates, key=lambda row: max(identity_drops(row)))
        maxima[f"{family}/{subfamily}"] = {"max_drop": max(identity_drops(worst)),
                                           "level": worst["level"],
                                           "mean_drop": worst["mean_drop"]}
    return maxima


# --------------------------------------------------------------------------
# Model-backed measurement
# --------------------------------------------------------------------------


def load_analysis(models_dir, provenance):
    from insightface.app import FaceAnalysis

    for filename, digest in provenance["extracted_models_sha256"].items():
        if sha256(models_dir / filename) != digest:
            raise ValueError(f"Model checksum mismatch: {filename}")
    app = FaceAnalysis(name=str(models_dir), allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    return app


def detect_and_embed(image, app):
    """Mirror score_faces.py: sort by left edge, then use the largest face."""
    from insightface.utils.face_align import norm_crop

    faces = sorted(app.get(image), key=lambda face: float(face.bbox[0]))
    if not faces:
        return {"n_faces": 0, "detected": False, "aligned": None, "cosine_input": None,
                "bbox": None, "kps": None, "detection_confidence": None}
    chosen = max(faces, key=lambda face: (float(face.bbox[2] - face.bbox[0])
                                          * float(face.bbox[3] - face.bbox[1]),
                                          -float(face.bbox[0])))
    aligned = norm_crop(image, landmark=chosen.kps, image_size=112)
    return {"n_faces": len(faces), "detected": True, "aligned": aligned,
            "cosine_input": np.asarray(chosen.normed_embedding, dtype=float),
            "image_width": int(image.shape[1]),
            "bbox": [float(value) for value in chosen.bbox],
            "kps": np.asarray(chosen.kps, dtype=float),
            "detection_confidence": float(chosen.det_score)}


def embed_crop(crop, recognition):
    """Embed an already-aligned 112x112 crop, bypassing detection and alignment."""
    feature = np.asarray(recognition.get_feat(crop), dtype=float).ravel()
    return feature / np.linalg.norm(feature)


def pose_centre(regime, base):
    """Point the portrait turns about: the landmark centroid of the face."""
    if regime == "end_to_end":
        return tuple(np.mean(np.asarray(base["kps"], dtype=float), axis=0))
    return tuple(ALIGNED_TEMPLATE_KPS.mean(axis=0))


def pose_focal_px(regime, base):
    """Focal length in pixels for a pose warp of one image in one regime.

    One camera focal for the photo. For the aligned crop, the same face keeps its
    focal-to-face-width ratio, so both regimes turn the face by the same amount.
    """
    focal = FOCAL_PER_FRAME_WIDTH * float(base["image_width"])
    if regime == "end_to_end":
        return focal
    face_width = float(base["bbox"][2] - base["bbox"][0])
    return focal * ALIGNED_CROP_SIZE / face_width


def measure(app, base_images, base_records, grid, seed=BASE_SEED):
    """Run both regimes over the frozen grid. Returns summaries and tiles.

    Occlusion landmarks stay in each image's own frame: the detected landmarks for
    a photo, the ArcFace template positions for an aligned crop. Getting this wrong
    makes every crop-regime occlusion vanish, so it is asserted by construction.
    """
    recognition = app.models["recognition"]
    base_embeddings = {identity: record["cosine_input"]
                       for identity, record in base_records.items()}
    summaries, tiles = [], {}
    for regime in ("end_to_end", "embed_only"):
        for entry in grid:
            per_identity = {}
            for identity in base_images:
                base = base_records[identity]
                focal_px = pose_focal_px(regime, base)
                centre = pose_centre(regime, base)
                if regime == "end_to_end":
                    perturbed = apply_perturbation(base_images[identity], entry,
                                                   kps=base["kps"], focal_px=focal_px,
                                                   centre=centre, seed=seed)
                    result = detect_and_embed(perturbed, app)
                    if result["detected"]:
                        value = cosine(base_embeddings[identity], result["cosine_input"])
                        tile = result["aligned"]
                    else:
                        value, tile = None, None
                    detected = result["detected"]
                    n_faces = result["n_faces"]
                else:
                    perturbed = apply_perturbation(base["aligned"], entry,
                                                   kps=ALIGNED_TEMPLATE_KPS,
                                                   focal_px=focal_px, centre=centre,
                                                   seed=seed)
                    value = cosine(base_embeddings[identity], embed_crop(perturbed, recognition))
                    detected, tile, n_faces = True, perturbed, 1
                per_identity[identity] = {"cosine": value, "detected": detected,
                                          "n_faces": n_faces}
                tiles[(regime, entry["level"], identity)] = tile
            summaries.append(summarise_level(regime, entry, per_identity))
            print(f"[{regime}] {entry['family']}/{entry['level']}: "
                  f"mean cosine {summaries[-1]['mean_cosine']}", flush=True)
    return summaries, tiles


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def write_curves_csv(path, summaries):
    fields = ["regime", "family", "subfamily", "level", "realistic", "mean_cosine",
              "std_cosine", "mean_drop", "std_drop", "identities_detected",
              "identities_scored", "identities_expected"]
    identities = sorted({identity for row in summaries for identity in row["per_identity_cosine"]})
    fields += [f"cosine_{identity}" for identity in identities]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            record = {field: row[field] for field in fields if field in row}
            for identity in identities:
                record[f"cosine_{identity}"] = row["per_identity_cosine"][identity]
            writer.writerow(record)


def write_verdicts_csv(path, classification):
    fields = ["photo_index", "month", "pass_index", "person_id", "face_index",
              "similarity", "best_other_identity_cosine", "margin",
              "drop_from_perfect", "status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in classification["rows"]:
            writer.writerow(row)


def write_iteration01_csv(path, check):
    fields = ["condition", "month", "pass1", "pass2", "drop", "exceeds_T",
              "exceeds_2T", "pass2_inside_envelope"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for condition, block in check["conditions"].items():
            for record in block["photos"]:
                writer.writerow({"condition": condition, **record})


def _label(canvas, text, origin, scale=0.32):
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)


def contact_sheet(path, regime, family, summaries, tiles, identities, tile_size=112,
                  label_height=30):
    """Grid of the exact aligned crops the embedder saw, one row per identity."""
    rows = [row for row in summaries if row["regime"] == regime and row["family"] == family]
    rows.sort(key=lambda row: row["level"])
    columns = len(rows) + 1
    sheet = np.full((label_height + len(identities) * (tile_size + label_height),
                     columns * (tile_size + 4) + 4, 3), 24, np.uint8)
    cv2.putText(sheet, f"{regime} / {family}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    for identity_index, identity in enumerate(identities):
        top = label_height + identity_index * (tile_size + label_height)
        left = 4
        base_tile = tiles.get((regime, "original", identity))
        if base_tile is not None:
            sheet[top:top + tile_size, left:left + tile_size] = cv2.resize(
                base_tile, (tile_size, tile_size))
        _label(sheet, identity.replace(IDENTITY_PREFIX, "")[:18], (left + 2, top + tile_size + 14))
        for column, row in enumerate(rows, start=1):
            left = 4 + column * (tile_size + 4)
            tile = tiles.get((regime, row["level"], identity))
            if tile is None:
                cv2.rectangle(sheet, (left, top), (left + tile_size, top + tile_size),
                              (60, 60, 60), -1)
                cv2.line(sheet, (left, top), (left + tile_size, top + tile_size), (0, 0, 255), 2)
                cv2.line(sheet, (left + tile_size, top), (left, top + tile_size), (0, 0, 255), 2)
                cv2.putText(sheet, "no face", (left + 8, top + tile_size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                sheet[top:top + tile_size, left:left + tile_size] = cv2.resize(
                    tile, (tile_size, tile_size))
            value = row["per_identity_cosine"].get(identity)
            colour = (140, 255, 140) if value is not None and value >= 0 else (120, 160, 255)
            _label(sheet, row["level"][:20], (left + 2, top + tile_size + 12), 0.3)
            _label(sheet, "n/a" if value is None else f"{value:.3f}",
                   (left + 2, top + tile_size + 26), 0.32)
            if not row["realistic"]:
                cv2.rectangle(sheet, (left, top), (left + tile_size, top + tile_size),
                              colour, 1)
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"Cannot write {path}")


def _fit_tile(canvas, image, left, top, size):
    """Place an image inside a square tile without changing its aspect ratio."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    offset_x = left + (size - resized.shape[1]) // 2
    offset_y = top + (size - resized.shape[0]) // 2
    canvas[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized


def full_frame_sheet(path, identity, summaries, frames, tile=224, columns=8, label_height=34):
    """Whole perturbed photos at their real aspect ratio, so pose realism can be eyeballed."""
    rows = sorted([row for row in summaries
                   if row["regime"] == "end_to_end" and row["family"] == "pose"],
                  key=lambda row: row["level"])
    cells = [(None, None)] + [(row["level"], row) for row in rows]
    bands = math.ceil(len(cells) / columns)
    sheet = np.full((label_height + bands * (tile + label_height), columns * (tile + 4) + 4, 3),
                    24, np.uint8)
    cv2.putText(sheet, f"end to end / pose / {identity}: whole warped photos",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    for index, (level, row) in enumerate(cells):
        band, column = divmod(index, columns)
        left = 4 + column * (tile + 4)
        top = label_height + band * (tile + label_height)
        if level is not None and frames.get(level) is not None:
            _fit_tile(sheet, frames[level], left, top, tile)
        value = None if row is None else row["per_identity_cosine"].get(identity)
        name = "baseline" if level is None else level
        _label(sheet, name[:26], (left + 2, top + tile + 14), 0.36)
        _label(sheet, "cosine 1.000" if row is None else
               ("no face" if value is None else f"cosine {value:.3f}"),
               (left + 2, top + tile + 30), 0.36)
        if row is not None and not row["realistic"]:
            cv2.rectangle(sheet, (left, top), (left + tile, top + tile), (120, 160, 255), 1)
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"Cannot write {path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT,
                        help="Root used for portable input hashes (default: repo root).")
    parser.add_argument("--images", type=Path,
                        help="Image directory with manifest.yaml (default: PROJECT/output/images).")
    parser.add_argument("--references", type=Path, nargs="*",
                        help="Reference portrait paths (default: from the manifest).")
    parser.add_argument("--scores", type=Path,
                        help="Frozen scores.json to check (default: evaluation/results/scores.json).")
    parser.add_argument("--iteration01", type=Path,
                        help="face_scores.csv from iteration 01 (default: single glob match).")
    parser.add_argument("--output", type=Path, default=HERE / "results_invariance",
                        help="Output directory (default: evaluation/results_invariance).")
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help="RNG seed recorded in the results (default: frozen seed).")
    parser.add_argument("--skip-sheets", action="store_true",
                        help="Skip PNG contact sheets.")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.images = (args.images or args.project_root / "output/images").resolve()
    args.scores = (args.scores or HERE / "results/scores.json").resolve()
    args.output = args.output.resolve()
    if not args.images.is_relative_to(args.project_root):
        parser.error("--images must be inside --project-root so input hashes remain portable.")
    if args.seed != BASE_SEED:
        parser.error(f"--seed is frozen at {BASE_SEED} so results stay comparable.")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    import yaml

    output = args.output
    crops_dir = output / "crops"
    sheets_dir = output / "contact_sheets"
    crops_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_sheets:
        sheets_dir.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((HERE / "model-provenance.json").read_text())
    models_dir = HERE / "model-cache/models/buffalo_l"
    inputs = {}
    manifest_path = args.images / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    inputs[manifest_path.relative_to(args.project_root).as_posix()] = sha256(manifest_path)
    scores_path = args.scores
    scores = json.loads(scores_path.read_text())
    inputs[scores_path.relative_to(args.project_root).as_posix()] = sha256(scores_path)

    if args.references:
        reference_paths = [path.resolve() for path in args.references]
    else:
        assets = {asset["entity_id"]: asset for asset in manifest["references"]}
        reference_paths = [args.images / assets[identity]["image"]["path"]
                           for identity in scores["matrix_row_order"]]
    identity_order = [path.stem.split("_person_")[-1] for path in reference_paths]
    identity_order = [f"{IDENTITY_PREFIX}{name}" for name in identity_order]

    app = load_analysis(models_dir, provenance)
    base_images, base_records, reference_rows = {}, {}, []
    for identity, path in zip(identity_order, reference_paths, strict=True):
        relative = path.relative_to(args.project_root).as_posix()
        inputs[relative] = sha256(path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        record = detect_and_embed(image, app)
        if record["n_faces"] != 1:
            raise ValueError(f"Reference {identity} has {record['n_faces']} faces; expected one.")
        base_images[identity], base_records[identity] = image, record
        crop_path = crops_dir / f"{identity}_aligned_112.png"
        if not cv2.imwrite(str(crop_path), record["aligned"]):
            raise OSError(f"Cannot write {crop_path}")
        inputs[crop_path.relative_to(args.project_root).as_posix()] = sha256(crop_path)
        reference_rows.append({
            "identity": identity, "source": relative, "source_sha256": inputs[relative],
            "aligned_crop": crop_path.relative_to(args.project_root).as_posix(),
            "aligned_crop_sha256": inputs[crop_path.relative_to(args.project_root).as_posix()],
            "bbox": record["bbox"],
            "landmarks": record["kps"].tolist(),
            "detection_confidence": record["detection_confidence"],
        })

    # The harness must reproduce the frozen baseline on unperturbed input.
    reference_embeddings = np.array([base_records[identity]["cosine_input"]
                                     for identity in identity_order])
    reference_cosine = reference_embeddings @ reference_embeddings.T
    baseline_check = {
        "reference_cosine_matrix": reference_cosine.tolist(),
        "scores_json_reference_cosine_matrix": scores["reference_cosine_matrix"],
        "reference_cosine_max_abs_diff": float(
            np.abs(reference_cosine - np.array(scores["reference_cosine_matrix"])).max()),
        "crop_vs_pipeline_embedding_max_abs_diff": float(max(
            np.abs(embed_crop(base_records[identity]["aligned"], app.models["recognition"])
                   - base_records[identity]["cosine_input"]).max()
            for identity in identity_order)),
    }
    scene_diff = 0.0
    for scene in scores["scenes"]:
        for passing in scene["passes"]:
            image = cv2.imread(str(args.images / passing["source"]))
            inputs[(args.images / passing["source"]).relative_to(args.project_root).as_posix()] = \
                sha256(args.images / passing["source"])
            faces = sorted(app.get(image), key=lambda face: float(face.bbox[0]))
            embeddings = [np.asarray(face.normed_embedding, dtype=float) for face in faces]
            if not embeddings:
                continue
            measured = reference_embeddings @ np.array(embeddings).T
            scene_diff = max(scene_diff, float(
                np.abs(measured - np.array(passing["cosine_matrix"])).max()))
    baseline_check["scene_cosine_max_abs_diff"] = scene_diff
    baseline_check["note"] = (
        "Harness embeddings against evaluation/results/scores.json. Differences are float32 "
        "storage in the frozen file, not a pipeline change.")

    grid = perturbation_grid()
    summaries, tiles = measure(app, base_images, base_records, grid, seed=args.seed)
    floor = select_floor(summaries)
    gap = between_identity_gap(scores["reference_cosine_matrix"], scores["matrix_row_order"])
    classification = classify_assignments(scores, floor)

    iteration01_path = args.iteration01
    if iteration01_path is None:
        matches = sorted(PROJECT.glob("experiments/iteration_01_*/results/face_scores.csv"))
        if len(matches) != 1:
            raise ValueError("Expected exactly one iteration_01 face_scores.csv; pass --iteration01.")
        iteration01_path = matches[0]
    iteration01_path = Path(iteration01_path).resolve()
    inputs[iteration01_path.relative_to(args.project_root).as_posix()] = sha256(iteration01_path)
    iteration01 = check_iteration01(parse_face_scores(iteration01_path.read_text()), floor)
    iteration01["source"] = iteration01_path.relative_to(args.project_root).as_posix()

    write_curves_csv(output / "curves.csv", summaries)
    write_verdicts_csv(output / "assignment_verdicts.csv", classification)
    write_iteration01_csv(output / "iteration01_check.csv", iteration01)

    sheet_paths = []
    if not args.skip_sheets:
        for regime in ("end_to_end", "embed_only"):
            for family in ("pose", "illumination", "resolution", "occlusion"):
                path = sheets_dir / f"{regime}_{family}.png"
                contact_sheet(path, regime, family, summaries, tiles, identity_order)
                sheet_paths.append(path.relative_to(output).as_posix())
        frames = {}
        for entry in grid:
            if entry["family"] != "pose":
                continue
            for identity in identity_order:
                warped = apply_perturbation(
                    base_images[identity], entry, kps=base_records[identity]["kps"],
                    focal_px=pose_focal_px("end_to_end", base_records[identity]),
                    centre=pose_centre("end_to_end", base_records[identity]),
                    seed=args.seed)
                frames[(identity, entry["level"])] = warped
        for identity in identity_order:
            path = sheets_dir / f"end_to_end_pose_frames_{identity}.png"
            full_frame_sheet(path, identity, summaries,
                             {level: frame for (key, level), frame in frames.items()
                              if key == identity})
            sheet_paths.append(path.relative_to(output).as_posix())

    result = {
        "purpose": "Empirical noise floor of the ArcFace cosine metric under identity-preserving nuisance.",
        "identity_threshold": None,
        "interpretation": (
            "Cosine similarity is not a probability and there is no calibrated identity threshold. "
            "These perturbations never change identity, so every measured change is metric "
            "sensitivity, not identity drift. T is an envelope for the nuisance families in the "
            "frozen grid only; families not modelled (expression, true 3D pose, age) can push a "
            "same-identity score lower, which makes T a lower bound and 'inside envelope' verdicts "
            "permissive."),
        "claim_under_test": iteration01["claim"],
        "seed": args.seed,
        "settings": {
            "model": "buffalo_l", "detector": "SCRFD-10GF",
            "recognition": "ArcFace ResNet50@WebFace600K",
            "provider": "CPUExecutionProvider", "det_size": [640, 640],
            "detection_threshold": 0.5, "aligned_crop_size": [112, 112],
            "multi_face_rule": "sort by bbox[0]; if several faces remain, use the largest bbox area, then the leftmost",
            "face_scale_interocular": FACE_SCALE_INTEROCULAR,
            "occlusion_fill_value": OCCLUSION_FILL,
            "pose_focal_per_frame_width": FOCAL_PER_FRAME_WIDTH,
            "pose_focal_px_per_identity": {
                regime: {identity: pose_focal_px(regime, record)
                         for identity, record in base_records.items()}
                for regime in ("end_to_end", "embed_only")},
            "resolution_note": (
                "Downscale sizes are short sides of the warped image. In the embed-only "
                "regime the aligned crop is already 112 px, so the 224/160/112 levels "
                "cannot downscale and are recorded as no-ops (cosine 1.0)."),
            "pose_model": "planar perspective warp of the image plane about the landmark centroid of the face; no 3D render, no self-occlusion",
        },
        "realistic_tier": {
            "rule": "identity preserved for a human observer under ordinary photographic variation",
            "pose_abs_deg_max": REALISTIC_POSE_MAX,
            "gamma_range": list(REALISTIC_GAMMA),
            "gain_range": list(REALISTIC_GAIN),
            "temperature_kelvin": list(REALISTIC_TEMPERATURE),
            "shading_strength_max": REALISTIC_SHADING_MAX,
            "resolution_short_side_min_px": REALISTIC_RESOLUTION_MIN,
            "blur_sigma_max": REALISTIC_BLUR_MAX,
            "jpeg_quality_min": REALISTIC_JPEG_MIN,
            "occlusion_area_fraction_max": REALISTIC_OCCLUSION_MAX,
            "excluded_from_T": "pose beyond 20 deg, gamma outside 0.7-1.4, gain outside 0.8-1.25, shading above 0.5, downscale below 160 px, blur above sigma 1, JPEG below q50, occlusion above 10%, sunglass and mask bands",
            "justification": "Each included level matches an ordinary photographic or rendering variation. Excluded levels remove the cues the embedder uses (eyes, mouth, fine detail) rather than re-rendering the same face, so folding them into T would inflate the floor until no real change could be detected.",
        },
        "grid": grid,
        "grid_counts": {family: sum(1 for entry in grid if entry["family"] == family)
                        for family in sorted({entry["family"] for entry in grid})},
        "packages": {name: version(name) for name in
                     ["insightface", "onnxruntime", "numpy", "opencv-python", "scipy"]},
        "model_provenance": provenance,
        "input_sha256": inputs,
        "references": reference_rows,
        "baseline_verification": baseline_check,
        "regimes": {
            "end_to_end": sorted([row for row in summaries if row["regime"] == "end_to_end"],
                                 key=lambda row: (row["family"], row["level"])),
            "embed_only": sorted([row for row in summaries if row["regime"] == "embed_only"],
                                 key=lambda row: (row["family"], row["level"])),
        },
        "floor": floor,
        "between_identity": {**gap, **signal_to_noise(floor, gap)},
        "realistic_family_maxima": realistic_family_maxima(summaries),
        "assignment_classification": classification,
        "iteration01": iteration01,
        "contact_sheets": sheet_paths,
        "runtime_seconds": time.perf_counter() - started,
        "generation_calls": 0,
    }
    (output / "invariance.json").write_text(
        json.dumps(result, indent=2, allow_nan=False, default=float) + "\n")
    print(f"Saved {output / 'invariance.json'}", flush=True)
    print(f"T = {floor['T']:.4f}  S_floor = {floor['S_floor']:.4f}  T_all = {floor['T_all']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
