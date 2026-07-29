import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils.utils import check_password
from Dividendos.core.ingestion.flex_query import import_flex_report
from Dividendos.core.ingestion.ibkr_flex_service import (
    fetch_flex_report,
    save_raw_xml,
    FlexServiceError,
)
from Dividendos.core.storage.db import (
    save_flex_import,
    read_import_log,
    save_target_allocation,
    read_target_allocation,
)

DB_PATH = Path(__file__).parent / "data" / "processed" / "dividendos.db"
RAW_DIR = Path(__file__).parent / "data" / "raw"


def _do_import(xml_text_or_path, source_label: str):
    parsed = import_flex_report(xml_text_or_path)
    return save_flex_import(DB_PATH, parsed, source_file=source_label)


def main():
    st.set_page_config(page_title="Importar - Dividendos", page_icon="⬆️", layout="wide")
    st.title("⬆️ Importar operaciones (cuenta de dividendos)")

    # -----------------------------------------------------------------
    # Opción principal: automático desde IBKR
    # -----------------------------------------------------------------
    st.markdown(
        "Descarga tu Flex Query directamente desde IBKR (rango: **Year to Date**). "
        "Reimportar no duplica datos."
    )

    if st.button("🔄 Actualizar desde IBKR", type="primary", use_container_width=True):
        try:
            ibkr_secrets = st.secrets["ibkr_dividendos"]
            token = ibkr_secrets["flex_token"]
            query_id = ibkr_secrets["flex_query_id"]
        except (KeyError, FileNotFoundError):
            st.error(
                "No encuentro `[ibkr_dividendos] flex_token` / `flex_query_id` "
                "en los Secrets de la app. Revisa Settings → Secrets en Streamlit Cloud."
            )
            return

        with st.spinner("Pidiendo el reporte a IBKR (puede tardar unos segundos)..."):
            try:
                xml_text = fetch_flex_report(token, query_id)
            except FlexServiceError as e:
                st.error(f"❌ IBKR devolvió un error ({e.code}): {e.message}")
                return
            except Exception as e:
                st.error(f"❌ Error de conexión con IBKR: {e}")
                return

        raw_path = save_raw_xml(xml_text, RAW_DIR)

        with st.spinner("Guardando en la base de datos..."):
            try:
                summary = _do_import(raw_path, source_label="IBKR (automático)")
            except Exception as e:
                st.error(f"❌ Error al procesar el XML recibido: {e}")
                return

        st.success(
            f"✅ Actualizado desde IBKR: **{summary['n_trades']}** operaciones, "
            f"**{summary['n_equity_rows']}** días de NLV, "
            f"**{summary['n_open_positions']}** posiciones abiertas, "
            f"**{summary['n_cash_transactions']}** movimientos de caja (dividendos incl.)."
        )

    st.markdown("---")

    # -----------------------------------------------------------------
    # Opción de respaldo: subida manual del XML
    # -----------------------------------------------------------------
    with st.expander("📎 Subir un XML manualmente (respaldo)"):
        st.caption(
            "Úsalo solo si el servicio automático falla, o si quieres cargar "
            "un histórico que ya no está dentro del rango de tu Flex Query."
        )
        uploaded_files = st.file_uploader(
            "Selecciona uno o varios archivos XML",
            type=["xml"],
            accept_multiple_files=True,
            key="xml_uploader",
        )

        if uploaded_files and st.button("📥 Importar archivo(s)", use_container_width=True):
            total_trades = total_equity = total_positions = total_tx = 0
            errors = []

            with st.spinner("Procesando archivo(s)..."):
                for uploaded_file in uploaded_files:
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        raw_path = RAW_DIR / f"{timestamp}_{uploaded_file.name}"
                        RAW_DIR.mkdir(parents=True, exist_ok=True)
                        raw_path.write_bytes(uploaded_file.getvalue())

                        summary = _do_import(raw_path, source_label=uploaded_file.name)
                        total_trades += summary["n_trades"]
                        total_equity += summary["n_equity_rows"]
                        total_positions += summary["n_open_positions"]
                        total_tx += summary["n_cash_transactions"]
                    except Exception as e:
                        errors.append(f"**{uploaded_file.name}**: {e}")

            if total_trades or total_equity:
                st.success(
                    f"✅ Importación manual completada: **{total_trades}** operaciones, "
                    f"**{total_equity}** días de NLV, **{total_positions}** posiciones, "
                    f"**{total_tx}** movimientos de caja."
                )
            for err in errors:
                st.error(f"❌ Error procesando {err}")

    st.markdown("---")

    # -----------------------------------------------------------------
    # Tabla de asignación objetivo (CSV/XLSX subido por el usuario)
    # -----------------------------------------------------------------
    st.markdown("### 🎯 Tabla de asignación objetivo")
    st.caption(
        "Sube tu tabla con columnas **Ticker** y **Target_%** (acepta variantes como "
        "'Target %', 'Objetivo %', etc.). Incluye una fila con Ticker = **CASH** si "
        "quieres fijar también el % objetivo de liquidez. Cada subida sustituye la "
        "tabla anterior por completo."
    )

    allocation_file = st.file_uploader(
        "Selecciona tu CSV o XLSX de asignación objetivo",
        type=["csv", "xlsx"],
        key="allocation_uploader",
    )

    if allocation_file is not None:
        try:
            if allocation_file.name.lower().endswith(".xlsx"):
                target_df = pd.read_excel(allocation_file)
            else:
                target_df = pd.read_csv(allocation_file)

            st.dataframe(target_df, use_container_width=True, hide_index=True)

            if st.button("💾 Guardar tabla de asignación objetivo", type="primary"):
                n = save_target_allocation(DB_PATH, target_df)
                st.success(f"✅ Guardados {n} tickers en la tabla de asignación objetivo.")
                st.rerun()
        except Exception as e:
            st.error(f"❌ No se pudo leer el archivo: {e}")

    current_target = read_target_allocation(DB_PATH) if DB_PATH.exists() else pd.DataFrame()
    if not current_target.empty:
        st.markdown("**Tabla de asignación objetivo actual guardada:**")
        st.dataframe(
            current_target,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": "Ticker",
                "target_pct": st.column_config.NumberColumn("% objetivo", format="%.2f%%"),
                "updated_at": "Actualizado",
            },
        )
    else:
        st.info("Todavía no has subido ninguna tabla de asignación objetivo.")

    st.markdown("---")

    # -----------------------------------------------------------------
    # Historial de importaciones
    # -----------------------------------------------------------------
    st.markdown("### 📋 Historial de actualizaciones")

    if DB_PATH.exists():
        log_df = read_import_log(DB_PATH)
        if not log_df.empty:
            st.dataframe(
                log_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": "ID",
                    "imported_at": "Importado el",
                    "source_file": "Origen",
                    "from_date": "Desde",
                    "to_date": "Hasta",
                    "n_trades": "Operaciones",
                    "n_equity_rows": "Días NLV",
                    "n_open_positions": "Posiciones abiertas",
                    "n_cash_transactions": "Mov. de caja",
                },
            )
        else:
            st.info("Todavía no se ha importado ningún reporte.")
    else:
        st.info("Todavía no se ha importado ningún reporte.")


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
