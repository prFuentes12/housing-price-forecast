import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
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
df = df[df['price'] > 0]  # Eliminar casas con precio 0

print(f"Valores nulos: {df.isnull().sum()}")
print(f"Valores duplicados: {df.duplicated().sum()}")
print(df.dtypes)

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

# Definir variables independientes (X) con TODAS las columnas numéricas excepto 'price'
X = df.drop(columns=['price'])
y = df['price']

# Dividir en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Entrenar modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 🔹 Predicciones
y_pred_rf = rf_model.predict(X_test)

# Función para evaluar modelos
def evaluar_modelo(y_test, y_pred, nombre):
    print(f"📌 Evaluación del modelo usando {nombre}:")
    print(f"   - MAE  (Error Absoluto Medio): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"   - RMSE (Raíz del Error Cuadrático Medio): {np.sqrt(root_mean_squared_error(y_test, y_pred)):.2f}")
    print(f"   - R² (Coeficiente de Determinación): {r2_score(y_test, y_pred):.4f}")
    print("-" * 50)

# Evaluar modelo Random Forest
evaluar_modelo(y_test, y_pred_rf, "Random Forest Regressor")

# 🔹 Validación Cruzada con Random Forest (5 folds)
cv_r2_rf = cross_val_score(rf_model, X, y, cv=5, scoring='r2', n_jobs=-1)
cv_mae_rf = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# Mostrar resultados de validación cruzada
print("\n📊 **Validación Cruzada (5 folds) para Random Forest**")
print(f"   - R² Promedio: {cv_r2_rf.mean():.4f} (Varianza: {cv_r2_rf.std():.4f})")
print(f"   - MAE Promedio: {-cv_mae_rf.mean():.2f} (Varianza: {cv_mae_rf.std():.2f})")

# 🔹 Importancia de las variables
importances = rf_model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# 📊 Gráfico de Importancia de Características
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'], palette='coolwarm')
plt.title('📌 Importancia de las Variables en Random Forest', fontsize=14)
plt.xlabel('Importancia')
plt.ylabel('Variables')
plt.show()
