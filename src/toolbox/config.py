"""Read JSON configuration; resolve relative paths against the project root."""

import json
from pathlib import Path

from toolbox.transforms import validate_transform


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def project_path(value, root):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def load_transform(path, key):
    """Read a bare 4x4 JSON array or an object containing the named transform."""
    value = load_config(path)
    if isinstance(value, dict):
        value = value[key]
    return validate_transform(value)
