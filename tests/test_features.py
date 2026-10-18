import importlib.util
import base64

import cv2
import numpy as np
import pytest


def features():
    assert importlib.util.find_spec('app.features'), 'Нужны локальные визуальные признаки'
    from app import features
    return features


def test_encoder_preprocessing_uses_rgb_and_imagenet_normalization():
    m = features()
    red = np.full((240, 320, 3), [0, 0, 255], dtype=np.uint8)
    tensor = m.preprocess(red)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor[0, :, 100, 100] == pytest.approx([2.2489083, -2.0357143, -1.8044444], abs=1e-5)


def test_sampled_features_keep_real_timestamps_and_cover_last_frame(tmp_path):
    m = features()
    path = tmp_path / 'sample.mov'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
    assert writer.isOpened()
    for i in range(60):
        writer.write(np.full((120, 160, 3), i, dtype=np.uint8))
    writer.release()

    class PixelEncoder:
        # Replace only the neural forward pass. Decode and timestamp logic remain real.
        def encode(self, image):
            return image.mean(axis=(0, 1))

    data = m.extract_video_features(path, PixelEncoder())
    assert data['times'] == pytest.approx([0, 2, 4, 5.9], abs=.05)
    assert data['features'].shape == (4, 9)
    assert np.isfinite(data['features']).all()


def test_onnx_can_load_from_a_windows_path_with_cyrillic_letters(tmp_path):
    m = features()
    assert hasattr(m, 'load_onnx'), 'Нужно чтение ONNX с кириллицей в пути'
    # Independently constructed tiny ONNX Relu model, no downloaded weights.
    raw = base64.b64decode('CAM6VAoSCgF4EgF5GgRyZWx1IgRSZWx1EgR0ZXN0WhsKAXgSFgoUCAESEAoCCAEKAggDCgIIAgoCCAJiGwoBeRIWChQIARIQCgIIAQoCCAMKAggCCgIIAkICEAc=')
    path = tmp_path / 'русская-папка' / 'тест.onnx'
    path.parent.mkdir()
    path.write_bytes(raw)
    net = m.load_onnx(path)
    net.setInput(-np.ones((1, 3, 2, 2), dtype=np.float32))
    assert np.array_equal(net.forward(), np.zeros((1, 3, 2, 2)))


def test_padding_in_mov_frame_count_uses_the_last_real_decodable_frame(tmp_path, monkeypatch):
    m = features()
    path = tmp_path / 'padded.mov'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
    for i in range(60):
        writer.write(np.full((120, 160, 3), i, dtype=np.uint8))
    writer.release()
    actual_capture = cv2.VideoCapture

    class PaddedMetadata:
        def __init__(self, filename):
            self.capture = actual_capture(filename)
        def get(self, prop):
            value = self.capture.get(prop)
            return value + 2 if prop == cv2.CAP_PROP_FRAME_COUNT else value
        def __getattr__(self, name):
            return getattr(self.capture, name)

    class PixelEncoder:
        def encode(self, image):
            return image.mean(axis=(0, 1))

    monkeypatch.setattr(cv2, 'VideoCapture', PaddedMetadata)
    data = m.extract_video_features(path, PixelEncoder())
    assert data['times'] == pytest.approx([0, 2, 4, 5.9], abs=.05)
    assert data['end_adjustment_frames'] == 2


def test_mov_edit_list_negative_first_timestamp_is_normalized_to_video_origin(tmp_path, monkeypatch):
    m = features()
    path = tmp_path / 'edit-list.mov'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
    for i in range(30):
        writer.write(np.full((120, 160, 3), i, np.uint8))
    writer.release()
    actual_capture = cv2.VideoCapture
    class EditListCapture:
        def __init__(self, filename):
            self.capture = actual_capture(filename)
        def get(self, prop):
            value = self.capture.get(prop)
            if prop == cv2.CAP_PROP_POS_MSEC and self.capture.get(cv2.CAP_PROP_POS_FRAMES) == 1:
                return -100
            return value
        def __getattr__(self, name):
            return getattr(self.capture, name)
    class PixelEncoder:
        def encode(self, frame):
            return frame.mean(axis=(0, 1))
    monkeypatch.setattr(cv2, 'VideoCapture', EditListCapture)
    data = m.extract_video_features(path, PixelEncoder())
    assert data['times'][0] == 0
    assert data['times'][1] == pytest.approx(2, abs=.05)
