
import pandas as pd

archivo = "datos/residuos_generados_superservicios.csv"
datos = pd.read_csv(archivo)

# Seleccionar sitios de disposición final
disposicion = datos[
    datos["TIPO_DE_SITIO_DESTINO"] ==
    "1_Sitio de Dispocion Final"
].copy()

# Identificar los años disponibles por municipio
cobertura = disposicion.groupby(
    "MUNICIPIO_ÁREA_DE_PRESTACIÓN"
)["AÑO_DEL_CARGUE"].nunique()

# Municipios presentes en los nueve años
municipios_completos = cobertura[cobertura == 9].index

print("\n========== COBERTURA DE MUNICIPIOS ==========")
print("Municipios con información completa:", len(municipios_completos))
print("Municipios con información incompleta:", (cobertura < 9).sum())

# Filtrar los municipios completos
datos_completos = disposicion[
    disposicion["MUNICIPIO_ÁREA_DE_PRESTACIÓN"].isin(
        municipios_completos
    )
]

# Calcular toneladas por año
resumen = datos_completos.groupby("AÑO_DEL_CARGUE")[
    "TONELADAS_DE_RESIDUOS_NO_APROVECHABLES"
].sum()

print("\n========== TONELADAS EN MUNICIPIOS COMPLETOS ==========")
print(resumen.to_string())

print("\n========== REGISTROS POR AÑO ==========")
print(datos_completos.groupby("AÑO_DEL_CARGUE").size().to_string())

print("\nRevisión terminada.")

