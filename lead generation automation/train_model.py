import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Generate dummy data
np.random.seed(42)
num_records = 100

# Random names
names = [f'Customer_{i}' for i in range(num_records)]

# Random interaction counts between 1 and 100
interaction_count = np.random.randint(1, 101, num_records)

# Random page views between 1 and 200
page_views = np.random.randint(1, 201, num_records)

# Random time on site between 1 and 1000 seconds
time_on_site = np.random.randint(1, 1001, num_records)

# Random conversion status (0 or 1)
converted = np.random.randint(0, 2, num_records)

# Create DataFrame
data = pd.DataFrame({
    'name': names,
    'interaction_count': interaction_count,
    'page_views': page_views,
    'time_on_site': time_on_site,
    'converted': converted
})

# Save to CSV (optional)
data.to_csv('dummy_leads.csv', index=False)

# Feature columns
feature_columns = ['interaction_count', 'page_views', 'time_on_site']

# Target column
target_column = 'converted'

# Splitting features and target
X = data[feature_columns]
y = data[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
model.save_model('xgb_model.json')

# Save the scaler
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
