import streamlit as st
from pathlib import Path
from datetime import datetime

from utils.utils import check_password
from Cartera.core.ingestion.flex_query import import_flex_report
from Cartera.core.ingestion.ibkr_flex_service import (
    fetch_flex_report,
    save_raw_xml,
    FlexServiceError,
)
from Cartera.core.storage.db import save_flex_import, read_import_log

DB_PATH = Path(__file__).parent / "data" / "processed" / "cartera.db"
RAW_DIR = Path(__file__).parent / "data" / "raw"


def _do_import(xml_text_or_path, source_label: str):
    """Parsea y guarda en SQLite. Devuelve el dict-resumen de save_flex_import."""
    parsed = import_flex_report(xml_text_or_path)
    return save_flex_import(DB_PATH, parsed, source_file=source_label)


def main():
    st.set_page_config(page_title="Actualizar - Cartera", page_icon="⬆️", layout="wide")
    st.title("⬆️ Actualizar operaciones")

    # -----------------------------------------------------------------
    # Opción principal: automático desde IBKR
    # -----------------------------------------------------------------
    st.markdown(
        "Descarga tu Flex Query directamente desde IBKR (rango: **Year to Date**, "
        "según cómo tienes configurada la query). Reimportar no duplica datos."
    )

    if st.button("🔄 Actualizar desde IBKR", type="primary", use_container_width=True):
        try:
            ibkr_secrets = st.secrets["ibkr"]
            token = ibkr_secrets["flex_token"]
            query_id = ibkr_secrets["flex_query_id"]
        except (KeyError, FileNotFoundError):
            st.error(
                "No encuentro `[ibkr] flex_token` / `flex_query_id` en los "
                "Secrets de la app. Revisa Settings → Secrets en Streamlit Cloud."
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

        # Guardamos el XML descargado en data/raw/ para auditoría
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
            f"**{summary['n_open_positions']}** posiciones abiertas."
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
        )

        if uploaded_files and st.button("📥 Importar archivo(s)", use_container_width=True):
            total_trades = total_equity = total_positions = 0
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
                    except Exception as e:
                        errors.append(f"**{uploaded_file.name}**: {e}")

            if total_trades or total_equity:
                st.success(
                    f"✅ Importación manual completada: **{total_trades}** operaciones, "
                    f"**{total_equity}** días de NLV, "
                    f"**{total_positions}** posiciones abiertas."
                )
            for err in errors:
                st.error(f"❌ Error procesando {err}")

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
