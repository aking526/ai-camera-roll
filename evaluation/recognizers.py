"""Face embedders behind one interface for a controlled ArcFace-vs-AdaFace swap.

Detection and alignment are held fixed so that only the embedder changes:
`align_face` is the only alignment path, and every embedder receives the
identical 112x112 BGR crop from it. The two embedders differ only in network
weights and in the input normalisation each exported graph expects.
"""

import cv2
import numpy as np

ALIGNED_SIZE = 112
EMBEDDING_DIM = 512

# insightface's ArcFaceONNX.get_feat feeds blobFromImages(1/127.5, size, 127.5, swapRB=True),
# i.e. RGB in [-1, 1]. Declared per embedder so a mismatch is a visible config change.
ARCFACE_PREPROCESSING = {"channel_order": "RGB", "mean": 127.5, "std": 127.5}
# Validated for this export by the same-person / different-person checks in
# test_recognizers.py and ADAFACE_AB.md; the graph itself has no input Sub/Mul node,
# so all normalisation must happen outside the network.
ADAFACE_PREPROCESSING = {"channel_order": "RGB", "mean": 127.5, "std": 127.5}


def align_face(image, landmarks, image_size=ALIGNED_SIZE):
    """Single alignment path: warp a BGR image onto the 5-point similarity template."""
    from insightface.utils.face_align import norm_crop

    crop = norm_crop(image, landmark=landmarks, image_size=image_size)
    if crop.shape != (image_size, image_size, 3):
        raise ValueError(f"Alignment returned {crop.shape}; expected {(image_size, image_size, 3)}")
    return crop


class OnnxEmbedder:
    """One ONNX recognition graph run on CPU, embedding aligned crops to unit length."""

    def __init__(self, name, model_path, channel_order, mean, std):
        import onnxruntime

        self.name = name
        self.model_path = model_path
        self.channel_order = channel_order
        self.mean = float(mean)
        self.std = float(std)
        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"{name}: expected one input and one output, "
                            f"got {len(inputs)} and {len(outputs)}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        shape = list(inputs[0].shape)
        if shape[1:] != [3, ALIGNED_SIZE, ALIGNED_SIZE]:
            raise ValueError(f"{name}: input shape {shape} is not [batch, 3, 112, 112]")
        if list(outputs[0].shape)[1:] != [EMBEDDING_DIM]:
            raise ValueError(f"{name}: output shape {outputs[0].shape} is not [batch, 512]")

    @property
    def preprocessing(self):
        return {"channel_order": self.channel_order, "mean": self.mean, "std": self.std,
                "layout": "NCHW", "aligned_crop_size": [ALIGNED_SIZE, ALIGNED_SIZE],
                "range": f"[{-self.mean / self.std:.1f}, {(255.0 - self.mean) / self.std:.1f}]"}

    def blob(self, aligned_bgr):
        """NCHW float32 blob for one aligned crop."""
        return preprocess_blob(aligned_bgr, self.channel_order, self.mean, self.std)

    def forward(self, aligned_bgr):
        """Raw 512-d network output, before any re-normalisation."""
        raw = self.session.run([self.output_name], {self.input_name: self.blob(aligned_bgr)})[0]
        return np.asarray(raw, dtype=np.float32).reshape(-1)

    def embed(self, aligned_bgr):
        """L2-normalised 512-d embedding; cosine similarity is then a plain dot product."""
        raw = self.forward(aligned_bgr)
        if raw.shape != (EMBEDDING_DIM,):
            raise ValueError(f"{self.name}: expected {EMBEDDING_DIM} values, got {raw.shape}")
        return l2_normalise(raw, self.name)


def preprocess_blob(aligned_bgr, channel_order, mean, std):
    """NCHW float32 blob for one aligned 112x112 BGR crop.

    cv2.dnn.blobFromImages is the exact call insightface's ArcFaceONNX.get_feat makes,
    so the ArcFace arm is bit-comparable with the frozen baseline.
    """
    if channel_order not in ("RGB", "BGR"):
        raise ValueError(f"Unsupported channel order: {channel_order}")
    if aligned_bgr.shape != (ALIGNED_SIZE, ALIGNED_SIZE, 3):
        raise ValueError(f"Expected a {(ALIGNED_SIZE, ALIGNED_SIZE, 3)} crop, "
                         f"got {aligned_bgr.shape}")
    if aligned_bgr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGR crop, got {aligned_bgr.dtype}")
    return cv2.dnn.blobFromImages(
        [aligned_bgr], 1.0 / float(std), (ALIGNED_SIZE, ALIGNED_SIZE),
        (float(mean),) * 3, swapRB=channel_order == "RGB")


def l2_normalise(raw, label="embedding"):
    """Scale a raw network output to unit length, rejecting degenerate vectors."""
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw.shape != (EMBEDDING_DIM,):
        raise ValueError(f"{label}: expected {EMBEDDING_DIM} values, got {raw.shape}")
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError(f"{label}: embedding norm {norm} is degenerate")
    embedding = (raw / norm).astype(np.float32)
    if abs(float(np.linalg.norm(embedding)) - 1.0) > 1e-5:
        raise ValueError(f"{label}: embedding is not unit length")
    return embedding


def load_arcface(model_path):
    return OnnxEmbedder("arcface", model_path, **ARCFACE_PREPROCESSING)


def load_adaface(model_path):
    return OnnxEmbedder("adaface", model_path, **ADAFACE_PREPROCESSING)
