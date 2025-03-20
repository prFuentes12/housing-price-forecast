import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("shree1992/housedata")
print("Path to dataset files:", path)


path2file = path +"/data.csv"
df = pd.read_csv(path2file)

#Valor nulos en cada columna
print(df.isnull().sum())

# Contar filas duplicadas
print(df.duplicated().sum())

#Comprobar columnas y sus tipos
print(df.dtypes)


#Muestra informacoión sobre las variables categóricas
categorical_columns = df.select_dtypes(include=['object']).columns  # Seleccionar solo columnas categóricas

for col in categorical_columns:
    print(f"{col}: {df[col].nunique()} valores únicos")


#City tiene un único valor, por lo que no aporta nada al dataset, se puede eliminar
df.drop(columns=['city'], inplace=True)