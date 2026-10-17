import importlib
import importlib.util
import shutil

import pytest


def dataset_module():
    assert importlib.util.find_spec("app.dataset") is not None, "Нужно написать app/dataset.py"
    return importlib.import_module("app.dataset")


@pytest.mark.parametrize("name", ["X-3.mov", "Х-3.mov", "х-3", "x-3.MOV"])
def test_latin_and_cyrillic_x_match_the_same_video(name):
    assert dataset_module().normalize_id(name) == "X-3"


@pytest.mark.parametrize("raw, expected", [
    (None, None), ("22 секунда", 22.0), ("1минута 8 секунда", 68.0),
    ("01:08", 68.0), (12.5, 12.5), ("5 секунда", 5.0),
])
def test_timestamps_from_labels_are_seconds(raw, expected):
    assert dataset_module().parse_timestamp(raw) == expected


@pytest.mark.parametrize("raw", ["неизвестно", "01:99", -2])
def test_invalid_timestamp_is_not_silently_changed_to_zero(raw):
    with pytest.raises(ValueError):
        dataset_module().parse_timestamp(raw)


def test_duplicate_movies_keep_both_labels_and_flag_different_timestamps(tmp_path):
    import cv2
    import numpy as np
    from openpyxl import Workbook

    first = tmp_path / "X-1.mov"
    writer = cv2.VideoWriter(str(first), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for _ in range(40):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    shutil.copyfile(first, tmp_path / "Х-2.mov")
    book = Workbook()
    sheet = book.active
    sheet.title = "labels"
    sheet.append(["наименование файла", "таймстемп", "нарушение / без нарушения"])
    sheet.append(["X-1", "1 секунда", "нарушение"])
    sheet.append(["Х-2", "2 секунда", "нарушение"])
    labels = tmp_path / "labels.xlsx"
    book.save(labels)
    manifest = dataset_module().build_manifest(tmp_path, labels)
    assert len(manifest["records"]) == 2
    assert [r["label_timecode_sec"] for r in manifest["records"]] == [1.0, 2.0]
    assert "duplicate_groups" in manifest, "Нужно явно сообщить о совпадающих файлах"
    assert manifest["duplicate_groups"][0]["different_timecodes"] is True
    assert manifest["duplicate_groups"][0]["conflicting_labels"] is False
