import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load the data
data = pd.read_csv('train_data.csv')

# Prepare features and target
X = data[['year', 'month', 'day', 'hour', 'minute']]
y = data['prediction']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Generate detailed performance metrics
performance_metrics = {
    'mean_squared_error': mse,
    'r2_score': r2,
    'feature_importances': dict(zip(X.columns, model.feature_importances_))
}

# Save performance metrics
with open('performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=4)

# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importances in Flight Prediction Model')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Visualization: Prediction vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual Predictions')
plt.xlabel('Actual Predictions')
plt.ylabel('Predicted Predictions')
plt.tight_layout()
plt.savefig('prediction_scatter.png')
plt.close()

# Visualization: Learning Curve
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
train_scores = []
test_scores = []

for size in train_sizes:
    X_subset = X_train_scaled[:int(len(X_train_scaled)*size)]
    y_subset = y_train[:int(len(y_train)*size)]
    
    model_subset = RandomForestRegressor(n_estimators=100, random_state=42)
    model_subset.fit(X_subset, y_subset)
    
    train_pred = model_subset.predict(X_subset)
    test_pred = model_subset.predict(X_test_scaled)
    
    train_scores.append(r2_score(y_subset, train_pred))
    test_scores.append(r2_score(y_test, test_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, label='Training Score', marker='o')
plt.plot(train_sizes, test_scores, label='Validation Score', marker='o')
plt.title('Learning Curves')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curves.png')
plt.close()

print("Model training and visualization complete!")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
