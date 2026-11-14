
import pandas as pd

archivo = "datos/serie_historica_superservicios.xlsx"

excel = pd.ExcelFile(archivo)
resultados = []

for hoja in excel.sheet_names:
    if "Diag." not in hoja and "Diag," not in hoja:
        continue

    año = int(hoja.replace("Diag,", "").replace("Diag.", "").strip())
    df = pd.read_excel(archivo, sheet_name=hoja)

    columna_anual = "TONELADAS ANUALES TOTALES"
    columna_diaria = "PROMEDIO T/DIA"
    columna_codigo = "COD DANE ATENDIDO"

    if columna_anual in df.columns:
        df[columna_anual] = pd.to_numeric(df[columna_anual], errors="coerce")

    if columna_diaria in df.columns:
        df[columna_diaria] = pd.to_numeric(df[columna_diaria], errors="coerce")

    total_anual = df[columna_anual].sum() if columna_anual in df.columns else None
    total_diario = df[columna_diaria].sum() if columna_diaria in df.columns else None
    municipios = df[columna_codigo].nunique() if columna_codigo in df.columns else 0

    resultados.append({
        "AÑO": año,
        "REGISTROS": len(df),
        "MUNICIPIOS_ATENDIDOS": municipios,
        "TOTAL_TONELADAS_ANUALES": total_anual,
        "SUMA_PROMEDIOS_DIARIOS": total_diario
    })

resumen = pd.DataFrame(resultados).sort_values("AÑO")

print("\n========== RESUMEN HISTÓRICO 2010-2024 ==========")
print(resumen.to_string(index=False))

resumen.to_csv(
    "resultados/resumen_historico_2010_2024.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo guardado correctamente.")
print("Análisis histórico terminado.")
