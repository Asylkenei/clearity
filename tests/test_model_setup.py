"""A corrupt download must never become an executable/model file."""
import hashlib
import importlib.util
from pathlib import Path

import pytest


def installer():
    assert importlib.util.find_spec('app.model_setup'), 'Нужна установка локальной модели'
    from app.model_setup import download_verified
    return download_verified


def test_download_rejects_bad_hash_and_removes_partial_file(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'not the expected model')
    target = tmp_path / 'models' / 'weights.gguf'
    with pytest.raises(ValueError, match='SHA256'):
        installer()(source.as_uri(), target, '0' * 64)
    assert not target.exists()
    assert not list(target.parent.glob('*.part'))


def test_download_reuses_only_verified_existing_file(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'valid weights')
    target = tmp_path / 'weights.gguf'
    target.write_bytes(b'broken')
    digest = hashlib.sha256(b'valid weights').hexdigest()
    installer()(source.as_uri(), target, digest)
    assert target.read_bytes() == b'valid weights'
    source.unlink()
    installer()(source.as_uri(), target, digest)
    assert target.read_bytes() == b'valid weights'
