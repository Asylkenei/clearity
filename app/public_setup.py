"""Отдельный ключ публичного доступа без изменения локального ключа."""

import secrets
from pathlib import Path

from dotenv import dotenv_values


def ensure_public_key(env_file: Path) -> str:
    values = dotenv_values(env_file)
    local_key = values.get("FORTE_API_KEY")
    if not local_key:
        raise RuntimeError("Сначала выполните setup.ps1: нужен локальный файл .env.")
    public_key = values.get("FORTE_PUBLIC_API_KEY")
    if public_key and public_key != local_key:
        return public_key
    public_key = secrets.token_urlsafe(32)
    with env_file.open("a", encoding="utf-8") as target:
        target.write("\nFORTE_PUBLIC_API_KEY=" + public_key + "\n")
    return public_key


if __name__ == "__main__":
    ensure_public_key(Path(__file__).resolve().parent.parent / ".env")
    print("Public access key is configured in .env (FORTE_PUBLIC_API_KEY).")
