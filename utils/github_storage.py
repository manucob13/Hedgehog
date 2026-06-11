# utils/github_storage.py
"""
Persistencia de calendars en GitHub via API.
Lee y escribe CSVs directamente en el repo.
Soporta múltiples archivos via parámetro csv_path.
"""
import base64
import io
import requests
import pandas as pd
import streamlit as st

# Path por defecto — mantiene compatibilidad con código existente que llama sin argumento
CSV_DEFAULT = "utils/data/calendars_TE.csv"


def _headers() -> dict:
    token = st.secrets["github"]["token"]
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _base_url(csv_path: str) -> str:
    owner = st.secrets["github"]["repo_owner"]
    repo  = st.secrets["github"]["repo_name"]
    # Si csv_path ya es una ruta completa (contiene "/") la usamos tal cual.
    # Si es solo un filename (ej. "calendars_TE_SPY.csv") lo ponemos en utils/data/
    if "/" in csv_path:
        path = csv_path
    else:
        path = f"utils/data/{csv_path}"
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"


def _branch() -> str:
    return st.secrets["github"].get("branch", "main")


# ==============================================================================
# LEER CSV DESDE GITHUB
# ==============================================================================

def cargar_calendars_csv(csv_path: str = CSV_DEFAULT) -> list[dict]:
    """
    Descarga el CSV de calendars desde GitHub y lo devuelve como lista de dicts.
    Si no existe el archivo devuelve lista vacía.

    Args:
        csv_path: nombre de archivo (ej. "calendars_TE_SPY.csv") o ruta completa
                  dentro del repo. Por defecto usa calendars_TE.csv.
    """
    try:
        resp = requests.get(
            _base_url(csv_path),
            headers=_headers(),
            params={"ref": _branch()},
            timeout=10,
        )
        if resp.status_code == 404:
            return []
        if resp.status_code != 200:
            st.warning(f"⚠️ No se pudo leer {csv_path} desde GitHub (HTTP {resp.status_code})")
            return []
        contenido = base64.b64decode(resp.json()["content"]).decode("utf-8")
        df = pd.read_csv(io.StringIO(contenido))
        return df.to_dict(orient="records")
    except Exception as e:
        st.warning(f"⚠️ Error cargando {csv_path} desde GitHub: {e}")
        return []


# ==============================================================================
# ESCRIBIR CSV EN GITHUB
# ==============================================================================

def guardar_calendars_csv(registros: list[dict], csv_path: str = CSV_DEFAULT) -> bool:
    """
    Sube la lista completa de registros como CSV a GitHub.
    Crea el archivo si no existe, lo actualiza si ya existe.

    Args:
        registros: lista de dicts a guardar.
        csv_path:  nombre de archivo o ruta completa dentro del repo.
                   Por defecto usa calendars_TE.csv.

    Returns:
        True si tuvo éxito.
    """
    try:
        df      = pd.DataFrame(registros)
        csv_str = df.to_csv(index=False)
        content_b64 = base64.b64encode(csv_str.encode("utf-8")).decode("utf-8")

        sha = _get_sha(csv_path)

        filename = csv_path.split("/")[-1]
        payload = {
            "message": f"Update {filename} ({len(registros)} registros)",
            "content": content_b64,
            "branch":  _branch(),
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(
            _base_url(csv_path),
            headers=_headers(),
            json=payload,
            timeout=15,
        )
        if resp.status_code in (200, 201):
            return True
        else:
            st.error(f"❌ Error guardando {filename} en GitHub (HTTP {resp.status_code}): {resp.text[:200]}")
            return False
    except Exception as e:
        st.error(f"❌ Excepción guardando en GitHub: {e}")
        return False


def _get_sha(csv_path: str = CSV_DEFAULT) -> str | None:
    """Obtiene el SHA del archivo actual (necesario para el PUT de actualización)."""
    try:
        resp = requests.get(
            _base_url(csv_path),
            headers=_headers(),
            params={"ref": _branch()},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("sha")
        return None
    except Exception:
        return None
