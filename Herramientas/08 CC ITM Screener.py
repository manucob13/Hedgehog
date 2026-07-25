"""
CC ITM Screener
================
Screener de Covered Calls IN-THE-MONEY (ITM) para ciclos cortos (Viernes -> Viernes).

    - DTE objetivo: 5 días de trading (7 días calendario, viernes a viernes)
    - Extrínseco objetivo: 0.90% - 1.00% del precio del subyacente
    - Delta objetivo del strike ITM: 0.60 - 0.75
    - Tendencia alcista: Close > SMA30 y SMA30 con pendiente positiva
    - Put/Call Ratio: usado como factor de ranking (no exclusión dura)
    - IV/RV ratio: prima "cara" vs. volatilidad realizada reciente
    - Liquidez: volumen subyacente, volumen/OI de opciones, spread bid-ask
    - Exclusión de earnings y ex-dividend dentro de la ventana del ciclo
    - Universo: Russell 1000 (descargado en vivo de iShares/IWB) + el
      universo curado de utils/tickers.py (Acciones + Índices + ETFs)

NOTA: la descarga de tickers del Russell 1000 se hace ahora con una función
simple y autocontenida (sin depender de utils/r1000_tickers.py), porque el
endpoint de BlackRock cambió de formato y rompía el parser anterior.
"""

import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
from statistics import NormalDist
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
import plotly.graph_objects as go
import plotly.express as px

from utils.utils import check_password
from utils.tickers import create_tickers_universe

try:
    from utils.cboe_utils import get_option_chain_cboe
    CBOE_AVAILABLE = True
except Exception:
    CBOE_AVAILABLE = False

warnings.filterwarnings('ignore')

_yfinance_lock = Lock()
_N = NormalDist()

RISK_FREE_RATE = 0.045


# ======================================================================
# 0. UNIVERSO DE TICKERS (versión simplificada, sin utils/r1000_tickers.py)
# ======================================================================

IWB_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/"
    "product-data/api/v1/get-fund-document"
    "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares"
    "&locale=en_US&portfolioId=239707&component=fundDownload&userType=individual"
)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
}
ROW_RE = re.compile(r"<ss:Row[^>]*>(.*?)</ss:Row>", re.DOTALL)
CELL_RE = re.compile(r"<ss:Data[^>]*>(.*?)</ss:Data>", re.DOTALL)


def _clean_ticker(raw):
    if raw is None:
        return None
    t = str(raw).strip().upper().replace("&amp;", "&")
    if not t or t in ("-", "NAN", "N/A"):
        return None
    return t.replace(" ", "")


def download_r1000_tickers():
    """Descarga y extrae tickers de holdings de IWB (Russell 1000) vía regex (robusto a XML mal formado)."""
    try:
        resp = requests.get(IWB_URL, headers=REQUEST_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.content.decode("utf-8", errors="ignore")
        rows = ROW_RE.findall(text)
        if not rows:
            return [], False

        header_idx = None
        for i, row in enumerate(rows):
            cells = CELL_RE.findall(row)
            if cells and cells[0].strip().lower() == "ticker":
                header_idx = i
                break
        if header_idx is None:
            return [], False

        tickers = []
        for row in rows[header_idx + 1:]:
            cells = CELL_RE.findall(row)
            if cells:
                t = _clean_ticker(cells[0])
                if t:
                    tickers.append(t)
        tickers = sorted(set(tickers))
        return (tickers, True) if tickers else ([], False)
    except Exception:
        return [], False


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_full_universe():
    """Universo completo: Russell 1000 + adicionales (utils/tickers.py). Cacheado 6h."""
    r1000_tickers, r1000_ok = download_r1000_tickers()

    df_extra = create_tickers_universe()
    extra_tickers = (
        df_extra["Ticker"].astype(str).tolist()
        if isinstance(df_extra, pd.DataFrame) else list(df_extra)
    )
    extra_tickers = sorted({t for t in (_clean_ticker(x) for x in extra_tickers) if t})

    all_tickers = sorted(set(r1000_tickers) | set(extra_tickers)) if r1000_ok else extra_tickers

    meta = {
        "r1000_ok": r1000_ok,
        "r1000_count": len(r1000_tickers),
        "extra_count": len(extra_tickers),
        "total_count": len(all_tickers),
    }
    df_universe = pd.DataFrame({"Ticker": all_tickers})
    return df_universe, meta


def refresh_full_universe():
    """Fuerza una descarga nueva (limpia caché) y devuelve (df_universe, meta)."""
    get_full_universe.clear()
    return get_full_universe()


# ======================================================================
# 1. DESCARGA DE DATOS BASE
# ======================================================================

def download_daily_data(ticker, period="6mo"):
    """Descarga precios diarios (para SMA30, RV, precio actual, volumen)."""
    try:
        end = datetime.now() + timedelta(days=1)
        period_days = {'3mo': 90, '6mo': 180, '1y': 365}
        days = period_days.get(period, 180)
        start = end - timedelta(days=days)
        with _yfinance_lock:
            data = yf.download(
                ticker, start=start, end=end,
                interval="1d", auto_adjust=False,
                multi_level_index=False, progress=False
            )
        if data is None or data.empty or len(data) < 40:
            return None
        data.index = pd.to_datetime(data.index)
        return data
    except Exception:
        return None


# ======================================================================
# 2. TENDENCIA: SMA30 + pendiente
# ======================================================================

def calculate_sma_trend(close, period=30, slope_lookback=5):
    try:
        if len(close) < period + slope_lookback:
            return None, None, None
        sma = close.rolling(window=period).mean()
        sma_now = sma.iloc[-1]
        sma_prev = sma.iloc[-1 - slope_lookback]
        if pd.isna(sma_now) or pd.isna(sma_prev):
            return None, None, None
        price_now = close.iloc[-1]
        dist_pct = (price_now - sma_now) / sma_now * 100
        slope_positive = sma_now > sma_prev
        return round(float(sma_now), 2), round(float(dist_pct), 2), bool(slope_positive)
    except Exception:
        return None, None, None


# ======================================================================
# 3. VOLATILIDAD REALIZADA (para IV/RV ratio)
# ======================================================================

def calculate_realized_vol(close, window=10):
    try:
        if len(close) < window + 1:
            return None
        rets = np.log(close / close.shift(1)).dropna()
        rv = float(rets.iloc[-window:].std() * np.sqrt(252) * 100)
        return round(rv, 2)
    except Exception:
        return None


# ======================================================================
# 4. EARNINGS Y DIVIDENDOS
# ======================================================================

def get_next_earnings_date(stock):
    try:
        cal = stock.calendar
        if cal is not None and 'Earnings Date' in cal:
            ed = cal['Earnings Date']
            if isinstance(ed, list) and len(ed) > 0:
                return pd.to_datetime(ed[0]).date()
            if pd.notna(ed):
                return pd.to_datetime(ed).date()
        return None
    except Exception:
        return None


def get_next_ex_dividend_date(stock):
    try:
        cal = stock.calendar
        if cal is not None and 'Ex-Dividend Date' in cal:
            exd = cal['Ex-Dividend Date']
            if pd.notna(exd):
                exd = pd.to_datetime(exd).date()
                if exd > date.today():
                    return exd
        divs = stock.dividends
        if divs is not None and not divs.empty and len(divs) >= 2:
            dates = divs.tail(4).index
            intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            avg_interval = np.mean(intervals)
            est = (divs.index[-1] + pd.Timedelta(days=avg_interval)).date()
            if est > date.today():
                return est
        return None
    except Exception:
        return None


# ======================================================================
# 5. SELECCIÓN DE VENCIMIENTO
# ======================================================================

def select_target_expiration(expirations, target_dte_days=7, tolerance_days=3):
    try:
        today = date.today()
        best = None
        best_diff = None
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            dte = (exp_date - today).days
            if dte <= 0:
                continue
            diff = abs(dte - target_dte_days)
            if diff <= tolerance_days and (best_diff is None or diff < best_diff):
                best = (exp_str, dte)
                best_diff = diff
        return best if best else (None, None)
    except Exception:
        return None, None


# ======================================================================
# 6. BLACK-SCHOLES: DELTA DE LA CALL
# ======================================================================

def bs_call_delta(S, K, T_years, sigma, r=RISK_FREE_RATE):
    try:
        if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return None
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))
        return round(float(_N.cdf(d1)), 3)
    except Exception:
        return None


# ======================================================================
# 7. SELECCIÓN DEL STRIKE ITM + MÉTRICAS DE LA OPCIÓN
# ======================================================================

def find_itm_candidate(calls_df, current_price, dte_calendar,
                        extrinsic_min_pct, extrinsic_max_pct,
                        delta_min, delta_max):
    try:
        itm = calls_df[calls_df['strike'] < current_price].copy()
        if itm.empty:
            return None

        itm['bid'] = itm['bid'].fillna(0)
        itm['ask'] = itm['ask'].fillna(0)
        itm['mid'] = (itm['bid'] + itm['ask']) / 2
        itm = itm[itm['mid'] > 0]
        if itm.empty:
            return None

        itm['intrinsic'] = current_price - itm['strike']
        itm['extrinsic'] = itm['mid'] - itm['intrinsic']
        itm['extrinsic_pct'] = itm['extrinsic'] / current_price * 100
        itm['spread_pct'] = np.where(
            itm['mid'] > 0, (itm['ask'] - itm['bid']) / itm['mid'] * 100, np.nan
        )

        T_years = dte_calendar / 365.0
        itm['iv'] = itm['impliedVolatility'].fillna(0)
        itm['delta'] = itm.apply(
            lambda row: bs_call_delta(current_price, row['strike'], T_years, row['iv']),
            axis=1
        )

        candidates = itm[
            (itm['extrinsic_pct'] >= extrinsic_min_pct) &
            (itm['extrinsic_pct'] <= extrinsic_max_pct) &
            (itm['delta'].notna()) &
            (itm['delta'] >= delta_min) &
            (itm['delta'] <= delta_max)
        ]

        if candidates.empty:
            return None

        best = candidates.sort_values('extrinsic_pct', ascending=False).iloc[0]
        return best
    except Exception:
        return None


# ======================================================================
# 8. PUT/CALL RATIO
# ======================================================================

def get_pcr_yahoo(stock, exp_str, current_price, price_range_pct=10):
    try:
        chain = stock.option_chain(exp_str)
        lower = current_price * (1 - price_range_pct / 100)
        upper = current_price * (1 + price_range_pct / 100)

        calls = chain.calls
        calls_r = calls[(calls['strike'] >= lower) & (calls['strike'] <= upper)].copy()
        puts = chain.puts
        puts_r = puts[(puts['strike'] >= lower) & (puts['strike'] <= upper)].copy()

        def activity(df):
            if df.empty:
                return 0
            vol = df['volume'].fillna(0)
            oi = df['openInterest'].fillna(0) if 'openInterest' in df.columns else 0
            return float(np.where(vol > 0, vol, oi).sum())

        call_act = activity(calls_r)
        put_act = activity(puts_r)
        if call_act == 0:
            return None
        return round(put_act / call_act, 3)
    except Exception:
        return None


def get_pcr_cboe(ticker, exp_str):
    if not CBOE_AVAILABLE:
        return None
    try:
        df = get_option_chain_cboe(ticker)
        if df is None or df.empty:
            return None
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        df_exp = df[df['expiry'] == exp_date]
        if df_exp.empty:
            return None
        call_vol = df_exp[df_exp['opt_type'] == 'C']['volume'].fillna(0).sum()
        put_vol = df_exp[df_exp['opt_type'] == 'P']['volume'].fillna(0).sum()
        if call_vol == 0:
            return None
        return round(float(put_vol / call_vol), 3)
    except Exception:
        return None


def pcr_sentiment_label(pcr):
    if pcr is None:
        return "N/A"
    if pcr < 0.7:
        return "🟢 Muy Alcista"
    if pcr < 1.0:
        return "🟢 Alcista"
    if pcr < 1.3:
        return "🟡 Neutral"
    return "🔴 Bajista"


# ======================================================================
# 9. SCORING COMPUESTO
# ======================================================================

def _minmax_score(value, lo, hi, invert=False):
    if value is None or hi == lo:
        return 0
    v = max(lo, min(hi, value))
    score = (v - lo) / (hi - lo) * 100
    return round(100 - score, 1) if invert else round(score, 1)


def calculate_composite_score(extrinsic_pct, iv_rv_ratio, downside_protection_pct,
                               options_activity, vol_oi_ratio, dist_sma30_pct, pcr,
                               extrinsic_min, extrinsic_max):
    s_extrinsic = _minmax_score(extrinsic_pct, extrinsic_min, extrinsic_max)
    s_ivrv = _minmax_score(iv_rv_ratio, 0.8, 2.0)
    s_downside = _minmax_score(downside_protection_pct, 0, 10)
    s_volume = _minmax_score(np.log10(options_activity + 1) if options_activity else 0, 1, 5)
    s_voloi = _minmax_score(vol_oi_ratio, 0, 2)
    s_trend = _minmax_score(dist_sma30_pct, 0, 15)
    s_pcr = _minmax_score(pcr if pcr is not None else 1.3, 0.3, 1.3, invert=True)

    score = (
        s_extrinsic * 0.20 +
        s_ivrv * 0.20 +
        s_downside * 0.15 +
        s_volume * 0.15 +
        s_voloi * 0.10 +
        s_trend * 0.10 +
        s_pcr * 0.10
    )
    return round(score, 1)


def get_semaphore(score):
    if score >= 70:
        return "🟢"
    if score >= 50:
        return "🟡"
    return "🔴"


# ======================================================================
# 10. ANÁLISIS DE UN TICKER
# ======================================================================

def analyze_ticker_itm(ticker, params):
    try:
        data = download_daily_data(ticker, period="6mo")
        if data is None:
            return None

        close = data['Close']
        volume = data['Volume']
        current_price = float(close.iloc[-1])
        avg_volume = float(volume.iloc[-20:].mean())

        if avg_volume < params['min_underlying_volume']:
            return None
        if not (params['min_price'] <= current_price <= params['max_price']):
            return None

        sma30, dist_sma_pct, slope_positive = calculate_sma_trend(close, period=30)
        if params['apply_trend_filter']:
            if sma30 is None or current_price <= sma30:
                return None
            if params['require_slope_positive'] and not slope_positive:
                return None

        rv = calculate_realized_vol(close, window=10)

        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        exp_str, dte_calendar = select_target_expiration(
            expirations,
            target_dte_days=params['target_dte_days'],
            tolerance_days=params['dte_tolerance']
        )
        if exp_str is None:
            return None

        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()

        next_earnings = get_next_earnings_date(stock)
        if params['exclude_earnings'] and next_earnings is not None:
            if date.today() <= next_earnings <= (exp_date + timedelta(days=2)):
                return None

        next_exdiv = get_next_ex_dividend_date(stock)
        if params['exclude_exdiv'] and next_exdiv is not None:
            if date.today() <= next_exdiv <= exp_date:
                return None

        chain = stock.option_chain(exp_str)
        calls = chain.calls
        if calls is None or calls.empty:
            return None

        best = find_itm_candidate(
            calls, current_price, dte_calendar,
            params['extrinsic_min'], params['extrinsic_max'],
            params['delta_min'], params['delta_max']
        )
        if best is None:
            return None

        strike_volume = float(best.get('volume') or 0)
        strike_oi = float(best.get('openInterest') or 0)
        spread_pct = float(best.get('spread_pct')) if pd.notna(best.get('spread_pct')) else None

        if strike_oi < params['min_oi']:
            return None
        if strike_volume < params['min_option_volume'] and strike_oi < params['min_oi'] * 1.5:
            return None
        if spread_pct is not None and spread_pct > params['max_spread_pct']:
            return None

        vol_oi_ratio = round(strike_volume / strike_oi, 3) if strike_oi > 0 else 0

        iv_pct = round(float(best['iv']) * 100, 2) if pd.notna(best.get('iv')) else None
        iv_rv_ratio = round(iv_pct / rv, 3) if (iv_pct and rv and rv > 0) else None
        if params['min_iv_rv_ratio'] and (iv_rv_ratio is None or iv_rv_ratio < params['min_iv_rv_ratio']):
            return None

        premium = round(float(best['mid']), 2)
        strike_price = float(best['strike'])
        breakeven = round(current_price - premium, 2)
        downside_protection_pct = round((current_price - breakeven) / current_price * 100, 2)

        pcr = get_pcr_cboe(ticker, exp_str)
        pcr_source = "CBOE"
        if pcr is None:
            pcr = get_pcr_yahoo(stock, exp_str, current_price)
            pcr_source = "Yahoo (derivado)"
        if params['apply_pcr_filter'] and pcr is not None and pcr >= params['max_pcr']:
            return None

        extrinsic_pct = round(float(best['extrinsic_pct']), 2)
        intrinsic = round(float(best['intrinsic']), 2)
        delta = float(best['delta'])

        annualized_return_pct = round(extrinsic_pct * (365 / dte_calendar), 1) if dte_calendar > 0 else None

        score = calculate_composite_score(
            extrinsic_pct, iv_rv_ratio, downside_protection_pct,
            strike_volume + strike_oi, vol_oi_ratio, dist_sma_pct, pcr,
            params['extrinsic_min'], params['extrinsic_max']
        )

        return {
            'Semáforo': get_semaphore(score),
            'Ticker': ticker,
            'Score': score,
            'Precio': round(current_price, 2),
            'Strike': round(strike_price, 2),
            'Vencimiento': exp_str,
            'DTE_cal': dte_calendar,
            'Delta': delta,
            'Extrínseco_%': extrinsic_pct,
            'Intrínseco': intrinsic,
            'Prima': premium,
            'Breakeven': breakeven,
            'Downside_Prot_%': downside_protection_pct,
            'Retorno_Anualizado_%': annualized_return_pct,
            'IV_%': iv_pct,
            'RV_%': rv,
            'IV_RV_Ratio': iv_rv_ratio,
            'Vol_Strike': int(strike_volume),
            'OI_Strike': int(strike_oi),
            'Vol/OI': vol_oi_ratio,
            'Spread_%': round(spread_pct, 2) if spread_pct is not None else None,
            'Avg_Vol_Subyacente': int(avg_volume),
            'SMA30': sma30,
            'Dist_SMA30_%': dist_sma_pct,
            'SMA30_Subiendo': slope_positive,
            'PCR': pcr,
            'PCR_Fuente': pcr_source,
            'Sentiment': pcr_sentiment_label(pcr),
            'Next_Earnings': next_earnings.isoformat() if next_earnings else "N/A",
            'Next_ExDiv': next_exdiv.isoformat() if next_exdiv else "N/A",
        }
    except Exception:
        return None


# ======================================================================
# 11. EJECUCIÓN EN PARALELO
# ======================================================================

def run_screener(tickers, params, progress_bar, status_text):
    results = []
    total = len(tickers)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_ticker_itm, t, params): t for t in tickers}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"🔍 Analizando {completed}/{total}...")
            result = future.result()
            if result is not None:
                results.append(result)
    status_text.text(f"✅ Completado: {len(results)} candidatos encontrados")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    return df


# ======================================================================
# 12. GRÁFICOS
# ======================================================================

def plot_price_with_sma(ticker, sma_period=30):
    try:
        data = download_daily_data(ticker, period="6mo")
        if data is None or data.empty:
            return None
        sma = data['Close'].rolling(sma_period).mean()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name=ticker,
            increasing_line_color='#6daa45',
            decreasing_line_color='#dd6974'
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=sma,
            mode='lines', name=f'SMA{sma_period}',
            line=dict(color='#e8af34', width=2)
        ))
        fig.update_layout(
            title=f'{ticker} — Precio + SMA{sma_period}',
            template='plotly_dark',
            height=420,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        return fig
    except Exception:
        return None


# ======================================================================
# 13. INTERFAZ STREAMLIT
# ======================================================================

def main():
    st.set_page_config(page_title="CC ITM Screener", page_icon="🎯", layout="wide")

    if not check_password():
        st.stop()

    st.title("🎯 CC ITM Screener — ITM Covered Calls (Viernes a Viernes)")
    st.markdown(
        "**Screener de covered calls ITM, ciclo de 5 días de trading "
        "(entrada viernes, vencimiento viernes siguiente) · "
        "Extrínseco objetivo 0.90%-1.00% · Universo: Russell 1000 + ETFs + Índices**"
    )
    st.markdown("---")

    st.markdown("### 📂 Universo de Tickers")

    col1, col2 = st.columns([3, 1])
    with col2:
        actualizar_btn = st.button("🔄 Actualizar Tickers (Russell 1000 + Adicionales)",
                                    use_container_width=True, type="primary")

    if actualizar_btn:
        with st.spinner("Descargando holdings de IWB (Russell 1000) "
                         "y combinando con el universo adicional..."):
            df_universe, meta = refresh_full_universe()
            st.session_state['itm_universe_df'] = df_universe
            st.session_state['itm_universe_meta'] = meta

    if 'itm_universe_df' not in st.session_state:
        try:
            with st.spinner("Cargando universo de tickers (caché / Russell 1000 + adicionales)..."):
                df_universe, meta = get_full_universe()
                st.session_state['itm_universe_df'] = df_universe
                st.session_state['itm_universe_meta'] = meta
        except Exception as e:
            st.error(f"Error cargando el universo de tickers: {e}")
            st.stop()

    df_universe = st.session_state['itm_universe_df']
    meta = st.session_state['itm_universe_meta']
    tickers_list = df_universe['Ticker'].astype(str).tolist()

    with col1:
        if meta.get('r1000_ok'):
            st.success(
                f"✅ **{meta['total_count']:,} tickers** en el universo "
                f"(Russell 1000: **{meta['r1000_count']:,}** "
                f"+ Adicionales: **{meta['extra_count']:,}** desde utils/tickers.py, "
                f"deduplicados)"
            )
        else:
            st.warning(
                f"⚠️ No se pudo descargar el listado del Russell 1000 (IWB). Usando solo "
                f"el universo adicional: **{meta['total_count']:,} tickers**. "
                f"Prueba a pulsar 'Actualizar Tickers'."
            )

    st.session_state['itm_tickers_universe'] = tickers_list

    st.markdown("---")

    st.markdown("### ⚙️ Configuración del Screener")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📅 Timing del ciclo**")
        target_dte_days = st.slider("DTE objetivo (días calendario)", 5, 10, 7,
                                     help="Viernes a viernes = 7 días calendario (5 de trading)")
        dte_tolerance = st.slider("Tolerancia DTE (± días)", 0, 5, 2)
        exclude_earnings = st.checkbox("Excluir earnings dentro de la ventana", value=True)
        exclude_exdiv = st.checkbox("Excluir ex-dividend dentro de la ventana", value=True)

        st.markdown("**💰 Extrínseco objetivo**")
        extrinsic_min, extrinsic_max = st.slider(
            "Rango de extrínseco (% del subyacente)", 0.3, 2.0, (0.90, 1.00), step=0.05
        )

    with c2:
        st.markdown("**🎯 Selección de strike (ITM)**")
        delta_min, delta_max = st.slider("Rango de Delta", 0.40, 0.95, (0.60, 0.75), step=0.05)
        min_iv_rv_ratio = st.slider("IV/RV mínimo", 0.5, 2.5, 1.1, step=0.05,
                                     help="IV implícita / Volatilidad realizada 10d — prima 'cara'")

        st.markdown("**📈 Tendencia**")
        apply_trend_filter = st.checkbox("Aplicar filtro SMA30 (Close > SMA30)", value=True)
        require_slope_positive = st.checkbox("Exigir SMA30 con pendiente positiva", value=True)

    with c3:
        st.markdown("**💧 Liquidez**")
        min_underlying_volume = st.number_input("Volumen mínimo subyacente (20d avg)",
                                                  value=1_000_000, step=100_000)
        min_price, max_price = st.slider("Rango de precio del subyacente ($)", 5, 800, (20, 400))
        min_option_volume = st.number_input("Volumen mínimo del strike (día)", value=500, step=100)
        min_oi = st.number_input("Open Interest mínimo del strike", value=500, step=100)
        max_spread_pct = st.slider("Spread bid-ask máximo (%)", 1, 20, 5)

        st.markdown("**🗳️ Sentimiento (PCR)**")
        apply_pcr_filter = st.checkbox("Excluir si PCR ≥ umbral (filtro duro)", value=False,
                                        help="Recomendado dejar como factor de ranking, no exclusión")
        max_pcr = st.slider("Umbral máximo de PCR", 0.5, 2.0, 1.3, step=0.1,
                             disabled=not apply_pcr_filter)

    params = {
        'target_dte_days': target_dte_days,
        'dte_tolerance': dte_tolerance,
        'exclude_earnings': exclude_earnings,
        'exclude_exdiv': exclude_exdiv,
        'extrinsic_min': extrinsic_min,
        'extrinsic_max': extrinsic_max,
        'delta_min': delta_min,
        'delta_max': delta_max,
        'min_iv_rv_ratio': min_iv_rv_ratio,
        'apply_trend_filter': apply_trend_filter,
        'require_slope_positive': require_slope_positive,
        'min_underlying_volume': min_underlying_volume,
        'min_price': min_price,
        'max_price': max_price,
        'min_option_volume': min_option_volume,
        'min_oi': min_oi,
        'max_spread_pct': max_spread_pct,
        'apply_pcr_filter': apply_pcr_filter,
        'max_pcr': max_pcr,
    }

    with st.expander("📋 Resumen de filtros activos", expanded=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            st.info(f"DTE objetivo: **{target_dte_days} ± {dte_tolerance} días**")
            st.info(f"Extrínseco: **{extrinsic_min}% - {extrinsic_max}%**")
            st.info(f"Delta: **{delta_min} - {delta_max}**")
            st.info(f"IV/RV mínimo: **{min_iv_rv_ratio}**")
        with fc2:
            st.info(f"Volumen subyacente ≥ **{min_underlying_volume:,}**")
            st.info(f"OI strike ≥ **{min_oi:,}** · Spread ≤ **{max_spread_pct}%**")
            st.info(f"Earnings excluidos: **{exclude_earnings}** · Ex-div excluidos: **{exclude_exdiv}**")
            st.info(f"CBOE PCR disponible: **{CBOE_AVAILABLE}**")

    st.markdown("---")

    st.markdown("### 🚀 Ejecutar Escaneo")
    scan_btn = st.button("🎯 INICIAR ESCANEO CC ITM", type="primary", use_container_width=True,
                          disabled=len(st.session_state.get('itm_tickers_universe', [])) == 0)

    if scan_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        tickers = st.session_state['itm_tickers_universe']
        df_results = run_screener(tickers, params, progress_bar, status_text)
        progress_bar.empty()

        if not df_results.empty:
            st.session_state['itm_scan_results'] = df_results
            st.session_state['itm_scan_timestamp'] = datetime.now()
            st.success(f"✅ Escaneo completado: **{len(df_results)}** candidatos ITM encontrados "
                       f"sobre {len(tickers)} tickers analizados")
        else:
            st.warning("⚠️ Ningún ticker cumplió todos los filtros. Prueba relajando algún criterio.")
            if 'itm_scan_results' in st.session_state:
                del st.session_state['itm_scan_results']

    st.markdown("---")

    st.markdown("### 📈 Resultados")

    if 'itm_scan_results' in st.session_state:
        df_show = st.session_state['itm_scan_results']
        ts = st.session_state['itm_scan_timestamp']

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🟢 Candidatos óptimos (Score≥70)", len(df_show[df_show['Score'] >= 70]))
        k2.metric("🟡 Candidatos moderados", len(df_show[(df_show['Score'] >= 50) & (df_show['Score'] < 70)]))
        k3.metric("📊 Total candidatos", len(df_show))
        k4.metric("🕐 Último escaneo", ts.strftime("%H:%M:%S"))

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 Tabla Completa", "🎯 Top Candidatos", "📉 Gráficos"])

        with tab1:
            def color_score(val):
                if val >= 70:
                    return 'background-color: #1e3a1e; color: #6daa45'
                if val >= 50:
                    return 'background-color: #3a3a1e; color: #e8af34'
                return 'background-color: #3a1e1e; color: #dd6974'

            display_cols = [
                'Semáforo', 'Ticker', 'Score', 'Precio', 'Strike', 'Vencimiento', 'DTE_cal',
                'Delta', 'Extrínseco_%', 'Prima', 'Breakeven', 'Downside_Prot_%',
                'Retorno_Anualizado_%', 'IV_%', 'RV_%', 'IV_RV_Ratio',
                'Vol_Strike', 'OI_Strike', 'Vol/OI', 'Spread_%',
                'Dist_SMA30_%', 'SMA30_Subiendo', 'PCR', 'Sentiment', 'PCR_Fuente',
                'Next_Earnings', 'Next_ExDiv'
            ]
            display_cols = [c for c in display_cols if c in df_show.columns]
            st.dataframe(
                df_show[display_cols].style.map(color_score, subset=['Score']),
                use_container_width=True,
                height=500
            )
            csv = df_show.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Descargar CSV", csv, "cc_itm_screener.csv", "text/csv")

        with tab2:
            top = df_show[df_show['Score'] >= 70].head(10)
            if top.empty:
                st.info("No hay candidatos con Score ≥ 70. Revisa la tabla completa o relaja filtros.")
            else:
                for _, row in top.iterrows():
                    with st.expander(
                        f"{row['Semáforo']} **{row['Ticker']}** — Score {row['Score']} | "
                        f"Strike ${row['Strike']} | Extrínseco {row['Extrínseco_%']}% | "
                        f"Delta {row['Delta']}"
                    ):
                        st.markdown(f"""
| Métrica | Valor |
|---|---|
| 💰 Precio subyacente | **${row['Precio']}** |
| 🎯 Strike ITM | **${row['Strike']}** |
| 📅 Vencimiento | **{row['Vencimiento']}** ({row['DTE_cal']}d calendario) |
| 🔺 Delta | `{row['Delta']}` |
| 💵 Prima | **${row['Prima']}** |
| 🧮 Extrínseco | **{row['Extrínseco_%']}%** |
| 🛡️ Downside protection | **{row['Downside_Prot_%']}%** |
| 📈 Retorno anualizado (extrínseco) | **{row['Retorno_Anualizado_%']}%** |
| 📊 IV / RV | {row['IV_%']}% / {row['RV_%']}% → ratio `{row['IV_RV_Ratio']}` |
| 💧 Vol/OI del strike | {row['Vol_Strike']} / {row['OI_Strike']} (`{row['Vol/OI']}`) |
| 〰️ Spread | {row['Spread_%']}% |
| 📐 Distancia SMA30 | {row['Dist_SMA30_%']}% (subiendo: {row['SMA30_Subiendo']}) |
| 🗳️ PCR | {row['PCR']} — {row['Sentiment']} ({row['PCR_Fuente']}) |
| 📆 Próx. earnings / ex-div | {row['Next_Earnings']} / {row['Next_ExDiv']} |
""")

        with tab3:
            if df_show.empty:
                st.info("Ejecuta el escaneo primero.")
            else:
                selected_chart = st.selectbox("Selecciona ticker para ver el gráfico",
                                               options=df_show['Ticker'].tolist())
                if selected_chart:
                    fig_price = plot_price_with_sma(selected_chart)
                    if fig_price:
                        st.plotly_chart(fig_price, use_container_width=True)
                    else:
                        st.warning("No se pudo cargar el gráfico de precios.")

                fig_scores = px.bar(
                    df_show.sort_values('Score', ascending=True).tail(20),
                    x='Score', y='Ticker', orientation='h', color='Score',
                    color_continuous_scale=['#dd6974', '#e8af34', '#6daa45'],
                    range_color=[0, 100],
                    title='Scores CC ITM — Top 20',
                    template='plotly_dark', height=500,
                    labels={'Score': 'Score CC ITM', 'Ticker': ''}
                )
                fig_scores.add_vline(x=70, line_dash='dash', line_color='#6daa45',
                                      annotation_text="Óptimo (70)")
                fig_scores.add_vline(x=50, line_dash='dash', line_color='#e8af34',
                                      annotation_text="Moderado (50)")
                st.plotly_chart(fig_scores, use_container_width=True)
    else:
        st.info("👆 Configura los parámetros y pulsa **INICIAR ESCANEO** para analizar el universo.")

    st.markdown("---")
    with st.expander("ℹ️ Guía de la estrategia y del scoring", expanded=False):
        st.markdown("""
### Reglas de elegibilidad (deben cumplirse todas las activadas)
- **Liquidez subyacente**: volumen 20d ≥ mínimo, precio dentro del rango
- **Timing**: expiración más cercana al DTE objetivo (viernes a viernes, tolerancia configurable)
- **Extrínseco**: entre el rango objetivo (% del precio del subyacente)
- **Strike ITM**: strike < precio actual, con Delta dentro del rango objetivo
- **Liquidez de la cadena**: OI y volumen del strike, spread bid-ask máximo
- **IV/RV**: la IV implícita del strike debe superar la vol. realizada reciente por el ratio mínimo
- **Tendencia**: Close > SMA30, opcionalmente con SMA30 en pendiente positiva
- **Eventos**: earnings y ex-dividend excluidos si caen dentro de la ventana del ciclo
- **PCR**: por defecto solo pondera el score; puede activarse como filtro duro

### Score compuesto (0-100)
| Factor | Peso |
|---|---|
| Extrínseco % (dentro del rango) | 20% |
| IV/RV ratio | 20% |
| Downside protection % | 15% |
| Volumen de opciones (log) | 15% |
| Vol/OI ratio del strike | 10% |
| Distancia % sobre SMA30 | 10% |
| PCR (invertido — más bajo, mejor) | 10% |

### Semáforo
- 🟢 Score ≥ 70 → candidato óptimo
- 🟡 Score 50-69 → candidato moderado, revisar manualmente
- 🔴 Score < 50 → no recomendado
""")


if __name__ == "__main__":
    main()
