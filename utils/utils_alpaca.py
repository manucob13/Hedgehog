"""
utils_alpaca.py
================
Utilidades para descargar datos de OPCIONES desde Alpaca (Market Data API +
Trading API, vía alpaca-py), en sustitución de yfinance para la Fase 2 del
screener de covered calls deep ITM (cadena de opciones: vencimientos, bid,
ask, mid, open interest, IV).

ALCANCE:
Este módulo resuelve, todo vía Alpaca: (a) el precio en vivo del
subyacente, (b) qué vencimientos de opciones existen para un ticker,
(c) si esos vencimientos tienen cadencia semanal real (filtro fijo), y
(d) la cadena de calls/puts de un vencimiento EXACTO con bid/ask/OI/IV.
Earnings y dividendos siguen viniendo de yfinance en el fichero
principal — Alpaca no los expone en un formato equivalente y no hacía
falta cambiarlos.

CREDENCIALES
------------
Se leen de st.secrets con esta forma esperada en secrets.toml:

    [alpaca]
    api_key = "..."
    secret_key = "..."
    paper = true          # opcional, default true (no afecta a datos de
                           # opciones, solo a qué cuenta usa TradingClient)

Si tu secrets.toml usa otros nombres de clave, ajusta get_alpaca_clients().

FEED DE DATOS
-------------
Opciones: por defecto se usa el feed "indicative" (gratuito, no requiere
suscripción OPRA). Si tu cuenta de Alpaca tiene la suscripción de datos de
opciones OPRA activada y quieres bid/ask más precisos y profundos, cambia
DEFAULT_OPTIONS_FEED a OptionsFeed.OPRA más abajo.
Acciones (precio en vivo del subyacente): por defecto se usa el feed "iex"
(gratuito). Si tu cuenta tiene suscripción SIP, cambia DEFAULT_STOCK_FEED
a DataFeed.SIP para precios consolidados de todos los mercados.

VENCIMIENTO EXACTO, SIN TOLERANCIA
-----------------------------------
A diferencia de la versión anterior basada en yfinance (que buscaba el
vencimiento más cercano a un DTE objetivo dentro de una tolerancia en días),
aquí get_option_chain() exige un vencimiento EXACTO: si ese vencimiento no
existe para el ticker, no se hace ningún fallback a otro cercano. El
screener es semanal — el llamador (el fichero principal) es responsable de
verificar antes que el vencimiento exacto elegido está en la lista que
devuelve get_option_expirations(), y de aplicar has_weekly_options() como
filtro fijo antes de pedir la cadena.
"""

import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import ContractType, AssetStatus
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockLatestTradeRequest
from alpaca.data.enums import OptionsFeed, DataFeed

logger = logging.getLogger("cc_itm_screener")

# Feed de opciones gratuito por defecto. Cambiar a OptionsFeed.OPRA si la
# cuenta tiene la suscripción de datos de opciones OPRA activada.
DEFAULT_OPTIONS_FEED = OptionsFeed.INDICATIVE

# Feed de acciones gratuito por defecto (precio en vivo del subyacente).
# Cambiar a DataFeed.SIP si la cuenta tiene esa suscripción.
DEFAULT_STOCK_FEED = DataFeed.IEX

# Ventana hacia delante (días) en la que se listan vencimientos disponibles.
CONTRACTS_LOOKAHEAD_DAYS = 60

# Un ticker se considera "con opciones semanales" si, dentro de
# WEEKLY_WINDOW_DAYS días, hay al menos dos vencimientos consecutivos
# separados por WEEKLY_MAX_GAP_DAYS días o menos (un ticker solo-mensual
# tiene gaps de ~28-35 días; uno semanal, de ~7, con algún salto puntual a
# 14 por festivos).
WEEKLY_MAX_GAP_DAYS = 8
WEEKLY_WINDOW_DAYS = 45


# ======================================================================
# Credenciales / clientes — cacheados como recurso (no como datos)
# ======================================================================

@st.cache_resource(show_spinner=False)
def get_alpaca_clients():
    """Crea (y cachea a nivel de proceso) el TradingClient, el
    OptionHistoricalDataClient y el StockHistoricalDataClient de Alpaca a
    partir de st.secrets. Devuelve (trading_client, option_data_client,
    stock_data_client)."""
    try:
        creds = st.secrets["alpaca"]
        api_key = creds["api_key"]
        secret_key = creds["secret_key"]
        paper = bool(creds.get("paper", True))
    except Exception as e:
        raise RuntimeError(
            "No se han encontrado credenciales de Alpaca en st.secrets. "
            "Añade una sección [alpaca] con api_key y secret_key a tu "
            "secrets.toml (ver docstring de utils_alpaca.py)."
        ) from e

    trading_client = TradingClient(api_key, secret_key, paper=paper)
    option_data_client = OptionHistoricalDataClient(api_key, secret_key)
    stock_data_client = StockHistoricalDataClient(api_key, secret_key)
    return trading_client, option_data_client, stock_data_client


def _get_page_token(resp):
    return resp.next_page_token if hasattr(resp, "next_page_token") else resp.get("next_page_token")


def _get_contracts_list(resp):
    return resp.option_contracts if hasattr(resp, "option_contracts") else resp.get("option_contracts", [])


# ======================================================================
# Vencimientos disponibles + chequeo de "opciones semanales"
# ======================================================================

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_option_expirations(ticker):
    """Lista ordenada (asc) de fechas (date) de vencimiento de CALLS
    activos y operables para `ticker`, dentro de los próximos
    CONTRACTS_LOOKAHEAD_DAYS días. Pagina la respuesta de Alpaca hasta
    agotar next_page_token. Solo se piden CALLS porque los vencimientos
    son los mismos para calls y puts — pedir solo un lado basta para
    conocer el calendario y reduce el payload a la mitad.

    Cacheado 6h: el calendario de vencimientos disponibles no cambia
    intradía, igual criterio que get_earnings_and_dividend_info() en el
    fichero principal."""
    trading_client, _, _ = get_alpaca_clients()
    today = date.today()
    expirations = set()
    page_token = None
    try:
        while True:
            req = GetOptionContractsRequest(
                underlying_symbols=[ticker],
                status=AssetStatus.ACTIVE,
                type=ContractType.CALL,
                expiration_date_gte=today,
                expiration_date_lte=today + timedelta(days=CONTRACTS_LOOKAHEAD_DAYS),
                limit=1000,
                page_token=page_token,
            )
            resp = trading_client.get_option_contracts(req)
            for c in _get_contracts_list(resp):
                exp = c.expiration_date
                if isinstance(exp, str):
                    exp = datetime.strptime(exp, "%Y-%m-%d").date()
                expirations.add(exp)
            page_token = _get_page_token(resp)
            if not page_token:
                break
    except Exception as e:
        logger.warning(f"[{ticker}] error listando vencimientos en Alpaca: {e}")
        return []
    return sorted(expirations)


def has_weekly_options(expirations,
                        max_gap_days=WEEKLY_MAX_GAP_DAYS,
                        window_days=WEEKLY_WINDOW_DAYS):
    """True si, entre los vencimientos futuros dentro de `window_days`
    días, hay al menos dos consecutivos separados por `max_gap_days` días
    o menos. Filtro FIJO del escáner (no configurable en la UI, igual que
    bid>0/ask>0/spread≤50%): el escáner es semanal por diseño, así que un
    ticker sin cadencia semanal real no debe pasar aunque su único
    vencimiento disponible coincida por casualidad con la fecha objetivo
    de una semana concreta."""
    today = date.today()
    upcoming = sorted(e for e in expirations if today < e <= today + timedelta(days=window_days))
    if len(upcoming) < 2:
        return False
    gaps = [(b - a).days for a, b in zip(upcoming, upcoming[1:])]
    return min(gaps) <= max_gap_days


# ======================================================================
# Cadena de opciones (calls + puts) para un vencimiento EXACTO
# ======================================================================

def _contracts_for_expiration(ticker, expiration_date):
    """Metadatos de contrato (incluye open_interest) para AMBOS lados
    (calls y puts) del vencimiento exacto `expiration_date`. Paginado.
    Devuelve dict {symbol: OptionContract}."""
    trading_client, _, _ = get_alpaca_clients()
    contracts = {}
    page_token = None
    while True:
        req = GetOptionContractsRequest(
            underlying_symbols=[ticker],
            status=AssetStatus.ACTIVE,
            expiration_date=expiration_date,
            limit=1000,
            page_token=page_token,
        )
        resp = trading_client.get_option_contracts(req)
        for c in _get_contracts_list(resp):
            contracts[c.symbol] = c
        page_token = _get_page_token(resp)
        if not page_token:
            break
    return contracts


# Guarda el desglose diagnóstico de la última llamada a get_option_chain()
# (ver get_last_chain_diag() más abajo).
_last_chain_diag = {}


def get_option_chain(ticker, expiration_date, feed=DEFAULT_OPTIONS_FEED):
    """Cadena de opciones completa (calls y puts por separado) para el
    vencimiento EXACTO `expiration_date` (date o 'YYYY-MM-DD'). Sin
    fallback a otro vencimiento: si `expiration_date` no está realmente
    listado para el ticker, la cadena vendrá vacía y el llamador debe
    tratarlo como "sin datos" (igual que antes con chain.calls vacío).

    Devuelve (calls_df, puts_df) con columnas equivalentes a las que
    devolvía yfinance (`stock.option_chain(exp).calls/.puts`), para que
    find_deep_itm_candidate() y compute_pcr() en el fichero principal NO
    necesiten cambios:

        strike, bid, ask, openInterest, impliedVolatility, volume,
        contractSymbol

    Si algo falla o no hay contratos, devuelve (None, None).

    NOTA sobre 'volume': el endpoint de snapshots de opciones de Alpaca no
    expone volumen acumulado del día en esta versión de la API — se usa el
    tamaño de la última operación (latest_trade.size) como aproximación
    puramente informativa, NO es volumen total del día. No se usa en
    ningún filtro duro, solo se muestra en la tabla de resultados (igual
    que antes).

    DIAGNÓSTICO (get_last_chain_diag()): tras cada llamada, este módulo
    guarda un dict con el desglose de dónde se pierde el dato si el
    resultado sale corto o vacío: cuántos contratos hay listados
    (TradingClient), cuántos snapshots de mercado devuelve Alpaca para
    ellos, y de esos, cuántos traen bid/ask realmente cotizado (>0). Es
    habitual que con el feed "indicative" (gratuito) los contratos existan
    y aparezcan en el snapshot pero sin bid/ask real en strikes deep ITM
    poco negociados — eso se ve aquí como snapshots_total alto pero
    quotes_with_bid_ask bajo.
    """
    global _last_chain_diag
    if isinstance(expiration_date, str):
        exp_str = expiration_date
        exp_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
    else:
        exp_date = expiration_date
        exp_str = expiration_date.isoformat()

    diag = {
        "feed": feed.value if hasattr(feed, "value") else str(feed),
        "contracts_total": 0,
        "snapshots_total": 0,
        "quotes_with_bid_ask": 0,
        "calls_total": 0,
        "puts_total": 0,
    }

    try:
        contracts = _contracts_for_expiration(ticker, exp_date)
        diag["contracts_total"] = len(contracts)
        if not contracts:
            _last_chain_diag = diag
            return None, None

        _, option_data_client, _ = get_alpaca_clients()
        chain_req = OptionChainRequest(
            underlying_symbol=ticker,
            expiration_date=exp_str,
            feed=feed,
        )
        snapshots = option_data_client.get_option_chain(chain_req)
        diag["snapshots_total"] = len(snapshots) if snapshots else 0

        rows = []
        for symbol, snap in snapshots.items():
            contract = contracts.get(symbol)
            if contract is None:
                continue

            q = snap.latest_quote
            bid = float(q.bid_price) if (q is not None and q.bid_price is not None) else 0.0
            ask = float(q.ask_price) if (q is not None and q.ask_price is not None) else 0.0
            if bid > 0 and ask > 0:
                diag["quotes_with_bid_ask"] += 1

            iv = snap.implied_volatility
            iv = float(iv) if iv is not None else None

            t = snap.latest_trade
            vol_proxy = float(t.size) if (t is not None and t.size is not None) else 0.0

            oi_raw = contract.open_interest
            oi = float(oi_raw) if oi_raw not in (None, "") else 0.0

            rows.append({
                "contractSymbol": symbol,
                "strike": float(contract.strike_price),
                "bid": bid,
                "ask": ask,
                "openInterest": oi,
                "impliedVolatility": iv,
                "volume": vol_proxy,
                "_type": contract.type.value if hasattr(contract.type, "value") else str(contract.type),
            })

        if not rows:
            _last_chain_diag = diag
            return None, None

        df = pd.DataFrame(rows)
        calls_df = df[df["_type"] == "call"].drop(columns=["_type"]).reset_index(drop=True)
        puts_df = df[df["_type"] == "put"].drop(columns=["_type"]).reset_index(drop=True)
        diag["calls_total"] = len(calls_df)
        diag["puts_total"] = len(puts_df)
        _last_chain_diag = diag
        return calls_df, puts_df
    except Exception as e:
        logger.warning(f"[{ticker}] error obteniendo cadena de opciones en Alpaca "
                        f"para {exp_str}: {e}")
        diag["error"] = str(e)
        _last_chain_diag = diag
        return None, None


def get_last_chain_diag():
    """Desglose diagnóstico de la última llamada a get_option_chain()
    (ver docstring de esa función). Pensado para mostrarse en la UI de
    Consulta Individual cuando el candidato sale vacío, para saber si el
    problema es que no hay contratos, no hay snapshots, o hay snapshots
    pero sin bid/ask cotizado (típico del feed "indicative" en strikes
    deep ITM poco negociados)."""
    return dict(_last_chain_diag)


# ======================================================================
# Precio en vivo del subyacente
# ======================================================================

def get_live_price(ticker, fallback_price, feed=DEFAULT_STOCK_FEED):
    """Precio en vivo del subyacente vía la última operación (latest
    trade) de Alpaca, con fallback transparente a `fallback_price` (el
    cierre histórico ya calculado en Fase 1 con yfinance) si Alpaca no
    devuelve dato o la llamada falla — por ejemplo fuera de horario en
    cuentas sin overnight/extended hours, o un fallo puntual de red.

    No se cachea: a diferencia de vencimientos/contratos (que cambian
    poco intradía), el precio en vivo debe pedirse fresco cada vez que
    se construye un candidato en Fase 2, igual que hacía antes la versión
    basada en yfinance fast_info."""
    try:
        _, _, stock_data_client = get_alpaca_clients()
        req = StockLatestTradeRequest(symbol_or_symbols=ticker, feed=feed)
        trades = stock_data_client.get_stock_latest_trade(req)
        trade = trades.get(ticker) if trades else None
        if trade is not None and trade.price is not None and float(trade.price) > 0:
            return float(trade.price)
    except Exception as e:
        logger.warning(f"[{ticker}] precio en vivo de Alpaca falló, uso fallback histórico: {e}")
    return fallback_price
