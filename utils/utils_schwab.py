"""
Utilidades para interactuar con la API de Schwab.
- El access_token se lee directamente desde GitHub (utils/data/schwab_token.json)
- Sin refresh, sin caché — el notebook diario es el único responsable de renovar el token
"""

import streamlit as st
import requests
import base64
import json
from datetime import datetime, date, timedelta


# ==============================================================================
# LEER TOKEN DESDE GITHUB
# ==============================================================================

def _get_token_from_github() -> dict | None:
    """
    Descarga schwab_token.json desde GitHub y lo devuelve como dict.
    """
    try:
        owner  = st.secrets["github"]["repo_owner"]
        repo   = st.secrets["github"]["repo_name"]
        branch = st.secrets["github"].get("branch", "main")
        token  = st.secrets["github"]["token"]

        url  = f"https://api.github.com/repos/{owner}/{repo}/contents/utils/data/schwab_token.json"
        resp = requests.get(
            url,
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            params={"ref": branch},
            timeout=10,
        )
        if resp.status_code == 404:
            st.error("❌ No se encontró schwab_token.json en GitHub. Corré el notebook de renovación.")
            return None
        if resp.status_code != 200:
            st.error(f"❌ Error leyendo token desde GitHub (HTTP {resp.status_code})")
            return None

        contenido = base64.b64decode(resp.json()["content"]).decode("utf-8")
        return json.loads(contenido)

    except Exception as e:
        st.error(f"❌ Excepción leyendo token desde GitHub: {e}")
        return None


# ==============================================================================
# OBTENER ACCESS TOKEN — solo desde GitHub, sin refresh
# ==============================================================================

def _get_valid_access_token() -> str | None:
    """
    Lee el access_token directamente desde GitHub.
    Sin caché, sin refresh — eso lo hace el notebook diario.
    """
    token_data = _get_token_from_github()
    if not token_data:
        return None
    access_token = token_data["token"].get("access_token")
    if not access_token:
        st.error("❌ No se encontró access_token en el token de GitHub.")
        return None
    return access_token


# ==============================================================================
# CLIENTE HTTP
# ==============================================================================

class SchwabClient:
    BASE = "https://api.schwabapi.com/marketdata/v1"

    def _headers(self):
        token = _get_valid_access_token()
        if not token:
            return None
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def get_quote(self, symbol: str):
        headers = self._headers()
        if not headers:
            return _FakeResponse(401, {})
        try:
            resp = requests.get(
                f"{self.BASE}/quotes",
                headers=headers,
                params={"symbols": symbol, "fields": "quote"},
                timeout=10,
            )
            return _FakeResponse(resp.status_code, resp.json() if resp.status_code == 200 else {})
        except Exception as e:
            st.error(f"❌ Error get_quote({symbol}): {e}")
            return _FakeResponse(500, {})

    def get_option_chain(self, symbol: str, from_date=None, to_date=None):
        headers = self._headers()
        if not headers:
            return _FakeResponse(401, {})
        params = {"symbol": symbol}
        if from_date:
            params["fromDate"] = from_date.isoformat() if hasattr(from_date, "isoformat") else str(from_date)
        if to_date:
            params["toDate"]   = to_date.isoformat()   if hasattr(to_date,   "isoformat") else str(to_date)
        try:
            resp = requests.get(
                f"{self.BASE}/chains",
                headers=headers,
                params=params,
                timeout=15,
            )
            return _FakeResponse(resp.status_code, resp.json() if resp.status_code == 200 else {})
        except Exception as e:
            st.error(f"❌ Error get_option_chain({symbol}): {e}")
            return _FakeResponse(500, {})


class _FakeResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


# ==============================================================================
# connect_to_schwab — interfaz pública
# ==============================================================================

def connect_to_schwab(*args, **kwargs):
    """
    Lee el access_token desde GitHub y devuelve un SchwabClient listo para usar.
    El refresh del token es responsabilidad exclusiva del notebook diario.
    """
    access_token = _get_valid_access_token()
    if not access_token:
        st.error("❌ No se pudo obtener el access_token desde GitHub.")
        return None
    return SchwabClient()


# ==============================================================================
# Helpers
# ==============================================================================

def normalize_ticker(ticker: str) -> str:
    ticker_upper = ticker.upper().strip()
    return "$SPX" if ticker_upper == "SPX" else ticker_upper


def normalize_date(date_input):
    if isinstance(date_input, str):
        return datetime.strptime(date_input, "%Y-%m-%d").date()
    elif isinstance(date_input, datetime):
        return date_input.date()
    return date_input


def get_date_range_for_ticker(ticker, target_date):
    normalized = normalize_ticker(ticker)
    if normalized == "$SPX":
        return target_date - timedelta(days=1), target_date + timedelta(days=1)
    return target_date - timedelta(days=2), target_date + timedelta(days=2)


def get_schwab_credentials():
    """Mantiene compatibilidad con código existente."""
    try:
        api_key      = st.secrets["schwab"]["api_key"]
        app_secret   = st.secrets["schwab"]["app_secret"]
        redirect_uri = st.secrets["schwab"].get("redirect_uri", "https://127.0.0.1")
        return api_key, app_secret, redirect_uri
    except KeyError as e:
        st.error(f"❌ Falta configurar secrets de Schwab: {e}")
        return None, None, None


def get_current_price_schwab(client, ticker: str):
    try:
        if client is None:
            return None
        symbol   = normalize_ticker(ticker)
        response = client.get_quote(symbol)
        if response.status_code != 200:
            return None
        quote_data = response.json()
        if symbol in quote_data:
            quote = quote_data[symbol].get("quote", {})
            for field in ("lastPrice", "mark", "closePrice"):
                val = quote.get(field)
                if val is not None:
                    return float(val)
            bid = quote.get("bidPrice")
            ask = quote.get("askPrice")
            if bid and ask and bid > 0 and ask > 0:
                return float((bid + ask) / 2)
        return None
    except Exception:
        return None


def obtener_datos_opcion(client, ticker, strike, tipo, fecha_salida):
    try:
        if client is None:
            return None, None, None
        symbol             = normalize_ticker(ticker)
        fecha_normalizada  = normalize_date(fecha_salida)
        from_date, to_date = get_date_range_for_ticker(symbol, fecha_normalizada)
        response = client.get_option_chain(symbol, from_date=from_date, to_date=to_date)
        if response.status_code != 200:
            return None, None, None
        opciones   = response.json()
        option_map = opciones.get("callExpDateMap" if tipo == "CALL" else "putExpDateMap", {})
        if not option_map:
            return None, None, None
        fecha_str       = fecha_normalizada.strftime("%Y-%m-%d")
        fecha_key_match = next((k for k in option_map if k.startswith(fecha_str)), None)
        if not fecha_key_match:
            return None, None, None
        strikes_dict = option_map[fecha_key_match]
        available    = {}
        for k in strikes_dict:
            try:
                available[float(k)] = k
            except ValueError:
                continue
        if not available:
            return None, None, None
        closest_key    = available[min(available, key=lambda x: abs(x - float(strike)))]
        contratos_list = strikes_dict[closest_key]
        if not contratos_list:
            return None, None, None
        c    = contratos_list[0]
        bid  = c.get("bid", 0) or 0
        ask  = c.get("ask", 0) or 0
        mark = c.get("mark", 0) or 0
        mid  = (bid + ask) / 2 if bid > 0 and ask > 0 else mark
        if not mid:
            return None, None, None
        return mid, c.get("delta"), c.get("theta")
    except Exception:
        return None, None, None


def get_atm_strike_schwab(client, ticker, current_price, expiration_date):
    try:
        symbol     = normalize_ticker(ticker)
        target     = normalize_date(expiration_date)
        from_date, to_date = get_date_range_for_ticker(symbol, target)
        response   = client.get_option_chain(symbol, from_date=from_date, to_date=to_date)
        if response.status_code != 200:
            return None
        data    = response.json()
        exp_str = target.strftime("%Y-%m-%d")
        strikes = set()
        for map_type in ("callExpDateMap", "putExpDateMap"):
            for date_key, strikes_dict in data.get(map_type, {}).items():
                if date_key.startswith(exp_str):
                    for k in strikes_dict:
                        try:
                            strikes.add(float(k))
                        except ValueError:
                            continue
        return min(strikes, key=lambda x: abs(x - current_price)) if strikes else None
    except Exception:
        return None
