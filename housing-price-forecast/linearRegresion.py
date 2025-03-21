import kagglehub
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np


# Descargar el dataset
dataset_path = kagglehub.dataset_download("shree1992/housedata")
print("Path to dataset files:", dataset_path)

# Cargar los datos
data_file = dataset_path + "/data.csv"
df = pd.read_csv(data_file)

# Revisar valores nulos, duplicados y valores de 0 en 'price'
print(f"Casas con precio 0: {df[df['price'] == 0].shape[0]}")
#df = df[df['price'] > 0]  # Eliminar casas con precio 0

print(f"Valores nulos: {df.isnull().sum()}")
print(f"Valores duplicados: {df.duplicated().sum()}")

df.describe()

#BOxplots para mostrar información
# fig,ax = plt.subplots(1,3)
# sns.countplot(data = df , x = "bathrooms" , ax = ax[0])
# sns.countplot(data = df , x = "bedrooms", ax = ax[1])
# sns.countplot(data = df , x = "floors", ax = ax[2])
# plt.tight_layout()


# Eliminar columna 'country' (solo tiene un valor único)
df.drop(columns=['country'], inplace=True)

# Convertir 'date' a datetime y extraer el mes
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
df['month'] = df['date'].dt.month
df.drop(columns=['date'], inplace=True)

# Encoding de variables categóricas (Target Encoding)
df['street_encoded'] = df.groupby('street')['price'].transform('mean')
df.drop(columns=['street'], inplace=True)

df['statezip_encoded'] = df.groupby('statezip')['price'].transform('mean')
df.drop(columns=['statezip'], inplace=True)

df['city_encoded'] = df.groupby('city')['price'].transform('mean')
df.drop(columns=['city'], inplace=True)

# Normalizar variables con StandardScaler (opcional para regresión lineal)
scaler = StandardScaler()
df[['statezip_encoded', 'city_encoded']] = scaler.fit_transform(df[['statezip_encoded', 'city_encoded']])

# Generar matriz de correlación y seleccionar las 5 variables más correlacionadas
correlation_matrix = df.corr()
correlation_target = correlation_matrix['price'].abs().sort_values(ascending=False)
top_5_features = correlation_target[1:6]
print("Las 5 variables más correlacionadas con price son:")
print(top_5_features)

# Definir variables independientes (X) y dependiente (y)
X_sqft = df[['sqft_living']]
X_bath = df[['bathrooms']]

#selecciono las 5 más relacionadas
X_mult = df[['sqft_living', 'bathrooms', 'statezip_encoded', 'street_encoded', 'sqft_above']]
y = df['price']

# Dividir en entrenamiento y prueba (80% - 20%)
X_train_sqft, X_test_sqft, y_train, y_test = train_test_split(X_sqft, y, test_size=0.2, random_state=42)
X_train_bath, X_test_bath, _, _ = train_test_split(X_bath, y, test_size=0.2, random_state=42)
X_train_mult, X_test_mult, y_train_mult, y_test_mult = train_test_split(X_mult, y, test_size=0.2, random_state=42)

# Modelo con sqft_living
model_sqft = LinearRegression()
model_sqft.fit(X_train_sqft, y_train)
y_pred_sqft = model_sqft.predict(X_test_sqft)

# Modelo con bathrooms
model_bath = LinearRegression()
model_bath.fit(X_train_bath, y_train)
y_pred_bath = model_bath.predict(X_test_bath)

# Modelo con múltiples variables
model_mult = LinearRegression()
model_mult.fit(X_train_mult, y_train_mult)
y_pred_mult = model_mult.predict(X_test_mult)


# Función para evaluar los modelos
def evaluar_modelo(y_test, y_pred, nombre):
    print(f"📌 Evaluación del modelo usando {nombre}:")
    print(f"   - MAE  (Error Absoluto Medio): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"   - RMSE (Raíz del Error Cuadrático Medio): {np.sqrt(root_mean_squared_error(y_test, y_pred)):.2f}")
    print(f"   - R² (Coeficiente de Determinación): {r2_score(y_test, y_pred):.4f}")
    print("-" * 50)


# Evaluar modelos
evaluar_modelo(y_test, y_pred_sqft, "sqft_living")
evaluar_modelo(y_test, y_pred_bath, "bathrooms")
evaluar_modelo(y_test_mult, y_pred_mult, "Modelo con múltiples variables")

# 🔹 Validación Cruzada para cada modelo
cv_r2_sqft = cross_val_score(model_sqft, X_sqft, y, cv=5, scoring='r2')
cv_mae_sqft = cross_val_score(model_sqft, X_sqft, y, cv=5, scoring='neg_mean_absolute_error')

cv_r2_bath = cross_val_score(model_bath, X_bath, y, cv=5, scoring='r2')
cv_mae_bath = cross_val_score(model_bath, X_bath, y, cv=5, scoring='neg_mean_absolute_error')

cv_r2_mult = cross_val_score(model_mult, X_mult, y, cv=5, scoring='r2')
cv_mae_mult = cross_val_score(model_mult, X_mult, y, cv=5, scoring='neg_mean_absolute_error')

# Mostrar resultados de validación cruzada
print("\n📊 **Validación Cruzada (5 folds) para cada modelo**")

print(f"\n📌 Modelo sqft_living:")
print(f"   - R² Promedio: {cv_r2_sqft.mean():.4f} (Varianza: {cv_r2_sqft.std():.4f})")
print(f"   - MAE Promedio: {-cv_mae_sqft.mean():.2f} (Varianza: {cv_mae_sqft.std():.2f})")

print(f"\n📌 Modelo bathrooms:")
print(f"   - R² Promedio: {cv_r2_bath.mean():.4f} (Varianza: {cv_r2_bath.std():.4f})")
print(f"   - MAE Promedio: {-cv_mae_bath.mean():.2f} (Varianza: {cv_mae_bath.std():.2f})")

print(f"\n📌 Modelo con múltiples variables:")
print(f"   - R² Promedio: {cv_r2_mult.mean():.4f} (Varianza: {cv_r2_mult.std():.4f})")
print(f"   - MAE Promedio: {-cv_mae_mult.mean():.2f} (Varianza: {cv_mae_mult.std():.2f})")
