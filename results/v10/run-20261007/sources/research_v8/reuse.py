"""Чтение закреплённых артефактов; полный приватный parent не требуется."""
import hashlib
from pathlib import Path, PurePosixPath

def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _object(value: object, name: str) -> dict:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name}: ожидался объект со строковыми ключами")
    return value



def _path(root: Path, relative: str) -> Path:
    """Входные ссылки не могут выходить из закреплённого архива."""
    root = root.resolve()
    name = PurePosixPath(relative)
    if name.is_absolute() or not name.parts or any(part in ("..", ".") for part in name.parts) or "\\" in relative:
        raise ValueError("Некорректный относительный путь артефакта")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Ссылка артефакта выходит из архива или является symlink")
    return path


def _verify_files(root: Path, manifest: dict) -> None:
    for relative, item in manifest.items():
        entry = _object(item, relative)
        content = _path(root, relative).read_bytes()
        if len(content) != entry.get("bytes") or _sha(content) != entry.get("sha256"):
            raise ValueError(f"Изменился артефакт родительского прогона: {relative}")


