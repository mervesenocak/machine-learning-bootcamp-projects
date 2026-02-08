import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("insurance.csv")
df.head()
df.info()
df.describe()
sns.boxplot(x="smoker", y="charges", data=df)
plt.title("Sigara İçme Durumuna Göre Sigorta Masrafları")
plt.show()
sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df)
plt.title("BMI ve Sigorta Masrafları")
plt.show()
X = df.drop("charges", axis=1)
y = df["charges"]
categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
print("Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2:", r2_score(y_test, y_pred_lr))
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Random Forest")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2:", r2_score(y_test, y_pred_rf))
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Gerçek Masraflar")
plt.ylabel("Tahmin Edilen Masraflar")
plt.title("Gerçek vs Tahmin (Random Forest)")
plt.plot([0, max(y_test)], [0, max(y_test)], color="red")
plt.show()
model = rf_model.named_steps["model"]
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

importance_df.head(10)
sns.barplot(x="Importance", y="Feature", data=importance_df.head(10))
plt.title("En Önemli 10 Feature")
plt.show()
