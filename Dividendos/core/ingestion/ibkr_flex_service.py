"""
ibkr_flex_service.py
Descarga automática del reporte Flex Query directamente desde IBKR,
usando el Flex Web Service (token + query ID), sin tener que descargar
el XML a mano desde el portal.

Flujo (2 pasos, según la documentación de IBKR):
    1. SendRequest  -> pide a IBKR que genere el reporte. Devuelve un
                       'ReferenceCode' (el reporte no está listo al instante).
    2. GetStatement -> con ese ReferenceCode, se pide el reporte ya generado.
                       Si todavía no está listo, IBKR devuelve el error 1019
                       ("Statement generation in progress") y hay que
                       reintentar a los pocos segundos.

Límites a tener en cuenta (documentados por IBKR):
    - El token caduca al año de haberlo generado.
    - No se debe llamar con demasiada frecuencia (evitar loops de refresco
      automático o pulsar "Actualizar" repetidamente en poco tiempo).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import requests

SEND_REQUEST_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"


class FlexServiceError(Exception):
    """Error devuelto por IBKR al pedir o recoger un Flex Statement."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[IBKR Flex error {code}] {message}")


def _request_statement(token: str, query_id: str) -> tuple[str, str]:
    """
    Paso 1: pide a IBKR que genere el reporte.
    Devuelve (reference_code, get_statement_url).
    Lanza FlexServiceError si IBKR responde con un fallo (token inválido,
    query ID inválido, etc.).
    """
    resp = requests.get(
        SEND_REQUEST_URL,
        params={"t": token, "q": query_id, "v": "3"},
        timeout=30,
    )
    resp.raise_for_status()

    root = ET.fromstring(resp.text)

    status = root.findtext("Status")
    if status != "Success":
        error_code = root.findtext("ErrorCode", default="???")
        error_message = root.findtext("ErrorMessage", default="Error desconocido")
        raise FlexServiceError(error_code, error_message)

    reference_code = root.findtext("ReferenceCode")
    get_statement_url = root.findtext("Url")

    if not reference_code or not get_statement_url:
        raise FlexServiceError("???", "Respuesta de IBKR incompleta (sin ReferenceCode/Url)")

    return reference_code, get_statement_url


def _fetch_statement(
    token: str,
    reference_code: str,
    get_statement_url: str,
    max_attempts: int = 10,
    wait_seconds: int = 5,
) -> str:
    """
    Paso 2: recoge el reporte ya generado, reintentando si IBKR aún lo
    está preparando (ErrorCode 1019). Devuelve el XML crudo (texto).
    """
    for attempt in range(1, max_attempts + 1):
        resp = requests.get(
            get_statement_url,
            params={"q": reference_code, "t": token, "v": "3"},
            timeout=30,
        )
        resp.raise_for_status()

        if resp.text.lstrip().startswith("<FlexQueryResponse"):
            return resp.text

        root = ET.fromstring(resp.text)
        error_code = root.findtext("ErrorCode", default="???")
        error_message = root.findtext("ErrorMessage", default="Error desconocido")

        if error_code == "1019":
            if attempt < max_attempts:
                time.sleep(wait_seconds)
                continue

        raise FlexServiceError(error_code, error_message)

    raise FlexServiceError(
        "TIMEOUT",
        f"IBKR no generó el reporte tras {max_attempts} intentos "
        f"({max_attempts * wait_seconds}s en total). Inténtalo de nuevo en un rato.",
    )


def fetch_flex_report(token: str, query_id: str) -> str:
    """
    Descarga el reporte Flex Query completo desde IBKR (los 2 pasos).
    Devuelve el XML crudo (texto).
    """
    reference_code, get_statement_url = _request_statement(token, query_id)
    return _fetch_statement(token, reference_code, get_statement_url)


def save_raw_xml(xml_text: str, raw_dir: str | Path) -> Path:
    """Guarda el XML descargado en data/raw/ con timestamp, para auditoría."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = raw_dir / f"{timestamp}_ibkr_auto.xml"
    path.write_text(xml_text, encoding="utf-8")
    return path
