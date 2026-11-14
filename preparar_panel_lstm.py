
import pandas as pd

archivo = "datos/serie_historica_superservicios.xlsx"
resultados = []

for anio in range(2010, 2025):

    hoja = f"Diag. {anio}" if anio < 2024 else "Diag, 2024"
    df = pd.read_excel(archivo, sheet_name=hoja)

    # Identificar la columna del municipio
    posibles = [
        "MUNICPIO ATENDIDO",
        "MUNICPIO",
        "MUNICIPIO ATENDIDO",
        "MUNICIPIO"
    ]

    columna_municipio = next(
        (c for c in posibles if c in df.columns), None
    )

    datos = df[
        ["COD DANE ATENDIDO", "PROMEDIO T/DIA"]
    ].copy()

    datos["MUNICIPIO"] = (
        df[columna_municipio] if columna_municipio else ""
    )

    datos["DEPARTAMENTO"] = (
        df["DEPARTAMENTO ATENDIDO"]
        if "DEPARTAMENTO ATENDIDO" in df.columns
        else ""
    )

    datos["COD DANE ATENDIDO"] = pd.to_numeric(
        datos["COD DANE ATENDIDO"], errors="coerce"
    )

    datos["PROMEDIO T/DIA"] = pd.to_numeric(
        datos["PROMEDIO T/DIA"], errors="coerce"
    )

    datos = datos.dropna(
        subset=["COD DANE ATENDIDO", "PROMEDIO T/DIA"]
    )

    datos["COD DANE ATENDIDO"] = (
        datos["COD DANE ATENDIDO"].astype("int64").astype(str).str.zfill(5)
    )

    # Consolidar sitios por código DANE
    agrupado = datos.groupby(
        "COD DANE ATENDIDO", as_index=False
    ).agg({
        "MUNICIPIO": "first",
        "DEPARTAMENTO": "first",
        "PROMEDIO T/DIA": "sum"
    })

    agrupado["AÑO"] = anio
    resultados.append(agrupado)

    print(
        f"{anio}: {len(agrupado)} municipios consolidados"
    )

panel = pd.concat(resultados, ignore_index=True)

panel = panel[
    [
        "COD DANE ATENDIDO",
        "DEPARTAMENTO",
        "MUNICIPIO",
        "AÑO",
        "PROMEDIO T/DIA"
    ]
]

panel.to_csv(
    "resultados/panel_municipal_2010_2024.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== RESULTADO ==========")
print("Registros:", len(panel))
print("Municipios únicos:", panel["COD DANE ATENDIDO"].nunique())
print("Años:", panel["AÑO"].nunique())
print("\nPrimeros registros:")
print(panel.head(10).to_string(index=False))
print("\nARCHIVO GUARDADO:")
print("resultados/panel_municipal_2010_2024.csv")
