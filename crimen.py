"""# Función para parsear manualmente cada línea del CSV considerando comillas
def manual_csv_parser(line):
    in_quotes = False
    field = ''
    fields = []
    
    for char in line:
        if char == '"' and (not field or field[-1] != '\\'):  # Detecta una comilla que no está escapada
            in_quotes = not in_quotes  # Cambia el estado de estar o no dentro de comillas
        elif char == ',' and not in_quotes:  # Si encuentra una coma fuera de comillas, es un delimitador de campo
            fields.append(field.strip())
            field = ''
        else:
            field += char
    
    if field:  # Agregar el último campo
        fields.append(field.strip())
    
    return fields

# Función para convertir la cadena de geo_shape a una lista de puntos
def parse_geo_shape(geo_shape):
    # Eliminar los caracteres iniciales y finales para quedarse solo con la parte de las coordenadas
    start = geo_shape.find('[[')
    end = geo_shape.rfind(']]')
    coords_text = geo_shape[start+2:end]

    # Dividir en pares de coordenadas y limpiar los corchetes
    coords_pairs = coords_text.split('], [')
    coords = []
    for pair in coords_pairs:
        pair = pair.strip('[]')  # Elimina los corchetes sobrantes de cada par
        x, y = map(float, pair.split(', '))
        coords.append((x, y))
    return coords


# Función para convertir la cadena geo_point_2d en una tupla
def parse_geo_point(geo_point):
    x, y = map(float, geo_point.split(","))
    return (x, y)

def leer_cuadrantes(file_path):

    # Diccionario para almacenar los datos
    data_dict = {}

    # Abrir y leer el archivo
    with open(file_path, mode='r', encoding='utf-8') as file:
        headers = manual_csv_parser(file.readline())  # Leer y parsear los nombres de columnas
        for line in file:
            values = manual_csv_parser(line)  # Leer y parsear cada línea usando la nueva función
            row_id = int(values[0])
            # Crear un diccionario para cada fila usando los índices correctos
            row_dict = {
                'no_region': int(values[1]),
                'no_cuadran': int(values[2]),
                'zona': values[3],
                'geo_shape': parse_geo_shape(values[4]),
                'geo_point_2d': parse_geo_point(values[5]),
                'alcaldia': values[6],
                'sector': values[7],
                'clave_sect': int(values[8])
            }
            # Agregar el diccionario de la fila al diccionario principal usando el ID como clave
            data_dict[row_id] = row_dict
    
    return data_dict
            

# Path al archivo CSV
cuadrantes = leer_cuadrantes('cuadrantes.csv')
print(cuadrantes[0])
"""

"""import pandas as pd

# Cargar el archivo XLSX
file_path = 'delitosAltoImpactoFeb2024.xlsx'  # Asegúrate de cambiar 'tu_archivo.xlsx' por la ruta de tu archivo
df = pd.read_excel(file_path)

# Guardar el DataFrame como CSV con ';' como separador
csv_file_path = 'delitosAltoImpactoFeb2024.csv'  # Asegúrate de cambiar 'tu_archivo.csv' por la ruta donde quieras guardar el archivo
df.to_csv(csv_file_path, sep=';', index=False)
"""


import pandas as pd

# Cargar los archivos .xlsx
df1 = pd.read_excel('delitosAltoImpacto2023.xlsx', engine='openpyxl')
df2 = pd.read_excel('delitosAltoImpactoFeb2024.xlsx', engine='openpyxl')

# Combinar los DataFrames
df_combinado = pd.concat([df1, df2])

# Guardar en un archivo .csv con ; como separador
df_combinado.to_csv('delitosAltoImpacto2023a2024.csv', sep=';', index=False)
