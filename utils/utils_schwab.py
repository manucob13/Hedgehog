"""
Utilidades para interactuar con la API de Schwab.

Sincronización con GitHub:
- El notebook diario (Jupyter) renueva refresh_token + access_token y los sube a GitHub.
- Streamlit lee el access_token de GitHub. Si está vencido, usa el refresh_token
  de GitHub para pedir uno nuevo a Schwab y SOLO actualiza access_token + expires_at
  de vuelta en GitHub (nunca toca refresh_token).
"""

import streamlit as st
import requests
import base64
import json
import time
from datetime import datetime, date, timedelta


# ==============================================================================
# LEER / ESCRIBIR TOKEN EN GITHUB
# ==============================================================================

def _github_config():
    try:
        owner  = st.secrets["github"]["repo_owner"]
        repo   = st.secrets["github"]["repo_name"]
        branch = st.secrets["github"].get("branch", "main")
        token  = st.secrets["github"]["token"]
        return owner, repo, branch, token
    except KeyError as e:
        st.error(f"❌ Falta configurar secrets de GitHub: {e}")
        return None, None, None, None


def _get_token_from_github():
    """
    Descarga schwab_token.json desde GitHub.
    Devuelve (token_data, sha) o (None, None) si falla.
    El sha es necesario para poder sobreescribir el archivo después.
    """
    try:
        owner, repo, branch, gh_token = _github_config()
        if not owner:
            return None, None

        url  = f"https://api.github.com/repos/{owner}/{repo}/contents/utils/data/schwab_token.json"
        resp = requests.get(
            url,
            headers={
                "Authorization": f"token {gh_token}",
                "Accept": "application/vnd.github.v3+json",
            },
            params={"ref": branch},
            timeout=10,
        )
        if resp.status_code == 404:
            st.error("❌ No se encontró schwab_token.json en GitHub. Corré el notebook de renovación.")
            return None, None
        if resp.status_code != 200:
            st.error(f"❌ Error leyendo token desde GitHub (HTTP {resp.status_code})")
            return None, None

        data = resp.json()
        contenido = base64.b64decode(data["content"]).decode("utf-8")
        return json.loads(contenido), data["sha"]

    except Exception as e:
        st.error(f"❌ Excepción leyendo token desde GitHub: {e}")
        return None, None


def _save_token_to_github(token_data, sha):
    """
    Sube el token_data actualizado a GitHub, sobreescribiendo el archivo existente.
    """
    try:
        owner, repo, branch, gh_token = _github_config()
        if not owner:
            return False

        url = f"https://api.github.com/repos/{owner}/{repo}/contents/utils/data/schwab_token.json"
        gh_headers = {
            "Authorization": f"token {gh_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        content_b64 = base64.b64encode(json.dumps(token_data, indent=2).encode()).decode()
        payload = {
            "message": f"Refresh access_token (Streamlit) — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": content_b64,
            "branch":  branch,
            "sha":     sha,
        }
        resp = requests.put(url, headers=gh_headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            return True
        st.warning(f"⚠️ No se pudo guardar el access_token renovado en GitHub (HTTP {resp.status_code})")
        return False

    except Exception as e:
        st.warning(f"⚠️ Excepción guardando token en GitHub: {e}")
        return False


# ==============================================================================
# RENOVAR ACCESS TOKEN CON SCHWAB (usando refresh_token)
# ==============================================================================

def _get_credentials():
    try:
        api_key    = st.secrets["schwab"]["api_key"]
        app_secret = st.secrets["schwab"]["app_secret"]
        return api_key, app_secret
    except KeyError as e:
        st.error(f"❌ Falta configurar secrets de Schwab: {e}")
        return None, None


def _refresh_access_token(api_key, app_secret, refresh_token):
    credentials = base64.b64encode(f"{api_key}:{app_secret}".encode()).decode()
    try:
        resp = requests.post(
            "https://api.schwabapi.com/v1/oauth/token",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"❌ Error renovando access_token (HTTP {resp.status_code}): {resp.text[:200]}")
        return None
    except Exception as e:
        st.error(f"❌ Excepción renovando access_token: {e}")
        return None


# ==============================================================================
# OBTENER ACCESS TOKEN VÁLIDO
# ==============================================================================

def _get_valid_access_token():
    """
    Lee el token desde GitHub.
    - Si el access_token todavía tiene margen (>2 min), lo usa directamente.
    - Si está vencido, usa el refresh_token (también de GitHub) para pedir uno
      nuevo a Schwab, y guarda SOLO access_token + expires_at de vuelta en GitHub.
    - NUNCA modifica ni sobreescribe el refresh_token.
    """
    token_data, sha = _get_token_from_github()
    if not token_data:
        return None

    access_token = token_data["token"].get("access_token")
    expires_at   = token_data["token"].get("expires_at", 0)

    # Margen de seguridad de 2 minutos
    if access_token and time.time() < expires_at - 120:
        return access_token

    # Access token vencido (o casi) -> renovar con refresh_token
    api_key, app_secret = _get_credentials()
    if not api_key:
        return access_token  # mejor esfuerzo: devolver el viejo si no hay credenciales

    refresh_token = token_data["token"].get("refresh_token")
    if not refresh_token:
        st.error("❌ No hay refresh_token en GitHub.")
        return access_token

    new_tokens = _refresh_access_token(api_key, app_secret, refresh_token)
    if not new_tokens:
        if access_token:
            st.warning("⚠️ No se pudo renovar el access_token — usando el existente (puede estar vencido).")
            return access_token
        return None

    new_access_token = new_tokens.get("access_token")
    if not new_access_token:
        return access_token

    now = int(time.time())
    token_data["token"]["access_token"] = new_access_token
    token_data["token"]["expires_in"]   = new_tokens.get("expires_in", 1800)
    token_data["token"]["expires_at"]   = now + new_tokens.get("expires_in", 1800)
    # IMPORTANTE: no tocar token_data["token"]["refresh_token"]
    if "id_token" in new_tokens:
        token_data["token"]["id_token"] = new_tokens["id_token"]

    if sha:
        _save_token_to_github(token_data, sha)

    return new_access_token


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
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


# ==============================================================================
# connect_to_schwab — interfaz pública
# ==============================================================================

def connect_to_schwab(*args, **kwargs):
    """
    Obtiene un access_token válido (desde GitHub, renovando si es necesario)
    y devuelve un SchwabClient listo para usar.
    """
    access_token = _get_valid_access_token()
    if not access_token:
        st.error("❌ No se pudo obtener un access_token válido desde GitHub.")
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
