import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import shap

file_path = "housing.csv"  # Zamień na rzeczywistą ścieżkę do pliku
df = pd.read_csv(file_path, sep=',')

print("Podglad danych:")
print(df.head())
print("\nInformacje o danych:")
print(df.info())
print("\nPodstawowe statystyki:")
print(df.describe())


# # Wizualizacja rozkładów cech numerycznych
# for column in df.select_dtypes(include=['float64', 'int64']).columns:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(df[column], kde=True, bins=30)
#     plt.title(f"Rozkład wartości - {column}")
#     plt.show()
#
# # Wizualizacja wykresów pudełkowych
# for column in df.select_dtypes(include=['float64', 'int64']).columns:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(df[column])
#     plt.title(f"Wykres pudełkowy - {column}")
#     plt.show()

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Tworzenie dodatkowych cech
df_encoded['rooms_per_household'] = df_encoded['total_rooms'] / df_encoded['households']
df_encoded['bedrooms_per_room'] = df_encoded['total_bedrooms'] / df_encoded['total_rooms']
df_encoded['population_per_household'] = df_encoded['population'] / df_encoded['households']

# Macierz korelacji
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Macierz korelacji cech")
plt.show()

# Usuwanie cech o niskiej korelacji z median_house_value
threshold = 0.3
correlations = df_encoded.corr()['median_house_value'].abs()
low_corr_features = correlations[correlations < threshold].index
print("Cechy o niskiej korelacji z ceną:", list(low_corr_features))
df_encoded = df_encoded.drop(columns=low_corr_features)

# Przygotowanie danych do modelowania
X = df_encoded.drop('median_house_value', axis=1)
y = df_encoded['median_house_value']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(len(X_test))

# Regresja liniowa
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)

# Random Forest (optymalizacja hiperparametrów)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
print("Najlepsze parametry Random Forest:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\nModel: {model_name}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

# Wyniki
evaluate_model(y_test, y_pred_lr, "Regresja Liniowa")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Feature Importance (dla Random Forest)
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importances - Random Forest")
plt.show()

# Poprawienie wywołania explainer dla SHAP
explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

# Wyświetlenie wykresów SHAP
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar")
plt.show()

# feature_name = 'median_income'
#
# plt.figure(figsize=(10, 6))
# shap.dependence_plot(feature_name, shap_values[0], X_test)  # [0] dla RandomForestRegressor
# plt.show()


# explainer = shap.TreeExplainer(best_rf_model)

# shap_values = explainer.shap_values(X_test)


# print(shap_values.shape)

# feature_name = 'median_income'

# shap.dependence_plot(feature_name, shap_values, X_test)
# plt.show()
