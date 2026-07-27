import streamlit as st
from pathlib import Path
from datetime import datetime

from utils.utils import check_password
from Cartera.core.ingestion.flex_query import import_flex_report
from Cartera.core.storage.db import save_flex_import, read_import_log

# Rutas de datos de esta sección (autocontenidas dentro de Cartera/)
DB_PATH = Path(__file__).parent / "data" / "processed" / "cartera.db"
RAW_DIR = Path(__file__).parent / "data" / "raw"


def main():
    st.set_page_config(
        page_title="Importar operaciones",
        page_icon="⬆️",
        layout="wide"
    )

    st.title("⬆️ Importar operaciones (Flex Query IBKR)")
    st.markdown(
        "Sube aquí el XML generado por tu **Activity Flex Query** de IBKR "
        "(Performance & Reports → Flex Queries → Run). Puedes subir varios "
        "archivos a la vez; reimportar un período ya importado no duplica "
        "datos, solo los actualiza."
    )

    uploaded_files = st.file_uploader(
        "Selecciona uno o varios archivos XML",
        type=["xml"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📥 Importar", type="primary", use_container_width=True):
            RAW_DIR.mkdir(parents=True, exist_ok=True)

            total_trades = 0
            total_equity_rows = 0
            total_positions = 0
            errors = []

            with st.spinner("Procesando archivo(s)..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Guardamos el XML original en data/raw/ para auditoría,
                        # con timestamp por si subes el mismo nombre varias veces.
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        raw_path = RAW_DIR / f"{timestamp}_{uploaded_file.name}"
                        raw_path.write_bytes(uploaded_file.getvalue())

                        parsed = import_flex_report(raw_path)
                        summary = save_flex_import(
                            DB_PATH, parsed, source_file=uploaded_file.name
                        )

                        total_trades += summary["n_trades"]
                        total_equity_rows += summary["n_equity_rows"]
                        total_positions += summary["n_open_positions"]

                    except Exception as e:
                        errors.append(f"**{uploaded_file.name}**: {e}")

            if total_trades or total_equity_rows:
                st.success(
                    f"✅ Importación completada: **{total_trades}** operaciones, "
                    f"**{total_equity_rows}** días de NLV, "
                    f"**{total_positions}** posiciones abiertas guardadas."
                )

            for err in errors:
                st.error(f"❌ Error procesando {err}")

    st.markdown("---")
    st.markdown("### 📋 Historial de importaciones")

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
                    "source_file": "Archivo",
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
