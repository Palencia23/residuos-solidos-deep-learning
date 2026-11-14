
import pandas as pd

archivo = "datos/serie_historica_superservicios.xlsx"

excel = pd.ExcelFile(archivo)

print("\n========== HOJAS DEL ARCHIVO EXCEL ==========")
print(excel.sheet_names)

for hoja in excel.sheet_names:
    print(f"\n========== HOJA: {hoja} ==========")

    datos = pd.read_excel(
        archivo,
        sheet_name=hoja,
        nrows=5
    )

    print("\nCOLUMNAS:")
    print(datos.columns.tolist())

    print("\nPRIMERAS FILAS:")
    print(datos.to_string(index=False))

print("\nRevisión del Excel terminada.")
