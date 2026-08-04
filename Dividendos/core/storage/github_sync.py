"""
github_sync.py
Sincroniza un archivo local (la base de datos SQLite) con un repositorio
privado de GitHub, usando la API de Contents. Esto permite que los datos
persistan entre reinicios del contenedor de Streamlit Cloud, que no tiene
almacenamiento persistente propio: cada vez que el contenedor se reinicia,
el sistema de archivos local vuelve a ser exactamente lo que hay en el
repo de código (sin las bases de datos, que están en .gitignore).

Flujo de uso en las páginas Streamlit:
    - Al principio de main() de páginas que LEEN datos (Panel, Calendario,
      Operaciones, Asignación): llamar a download_db() para traer la
      última versión guardada en GitHub antes de leer la SQLite local.
    - Después de guardar datos nuevos (tras save_flex_import(),
      save_target_allocation()): llamar a upload_db() para subir la
      versión local actualizada al repo.
"""

from __future__ import annotations

import base64
from pathlib import Path

import requests

API_BASE = "https://api.github.com"


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _contents_url(repo: str, path: str) -> str:
    return f"{API_BASE}/repos/{repo}/contents/{path}"


def download_db(
    local_path: str | Path,
    remote_path: str,
    token: str,
    repo: str,
    branch: str = "main",
) -> bool:
    """
    Descarga `remote_path` del repo privado y lo guarda en `local_path`.

    Devuelve True si se descargó algo, False si el archivo remoto todavía
    no existe (primera vez que se usa la app, no es un error).
    Lanza una excepción (requests.HTTPError) si hay un problema real
    (token inválido, repo incorrecto, etc.).
    """
    resp = requests.get(
        _contents_url(repo, remote_path),
        headers=_headers(token),
        params={"ref": branch},
        timeout=30,
    )

    if resp.status_code == 404:
        return False

    resp.raise_for_status()
    data = resp.json()

    file_bytes = base64.b64decode(data["content"])

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(file_bytes)
    return True


def upload_db(
    local_path: str | Path,
    remote_path: str,
    token: str,
    repo: str,
    branch: str = "main",
    commit_message: str = "Actualizar base de datos",
) -> None:
    """
    Sube (crea o actualiza) `local_path` al repo privado en `remote_path`.
    """
    local_path = Path(local_path)
    content_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")

    # La API de GitHub exige el 'sha' del archivo actual para poder
    # actualizarlo (si no existe aún, se crea sin ese campo).
    existing = requests.get(
        _contents_url(repo, remote_path),
        headers=_headers(token),
        params={"ref": branch},
        timeout=30,
    )

    payload = {
        "message": commit_message,
        "content": content_b64,
        "branch": branch,
    }
    if existing.status_code == 200:
        payload["sha"] = existing.json()["sha"]
    elif existing.status_code != 404:
        existing.raise_for_status()

    resp = requests.put(
        _contents_url(repo, remote_path),
        headers=_headers(token),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
