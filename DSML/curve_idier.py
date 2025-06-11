import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Generate synthetic data
np.random.seed(42)
n_groups_per_curve = 10000
n_points_per_group = 10
noise_std = 0.02

def generate_group(curve_func, x_range=(0, 1)):
    x = np.random.uniform(x_range[0], x_range[1], n_points_per_group)
    y = curve_func(x) + np.random.normal(0, noise_std, n_points_per_group)
    return x, y

curves = [
    lambda x: x,        
    lambda x: x**1.1,  
    lambda x: x**0.9, 
    lambda x: np.cos(x), 
    lambda x: np.sin(x) 
]

# Generate training data
X_data = []  # Features: [SSE1, SSE2, SSE3]
y_data = []  # Labels: 0, 1, or 2

for label, curve_func in enumerate(curves):
    for _ in range(n_groups_per_curve):
        x, y = generate_group(curve_func)
        # Compute SSE for each curve
        sse = []
        for c_func in curves:
            y_pred = c_func(x)
            sse.append(np.sum((y - y_pred)**2))
        X_data.append(sse)
        y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data)

# Step 2: Split and standardize data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train the model
model = LogisticRegression(multi_class='multinomial', random_state=42)
model.fit(X_train_scaled, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Step 5: Save the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 6: Deploy the model (example with new points)
def classify_points(x, y, model, scaler, curves):
    # Compute features
    sse = []
    for c_func in curves:
        y_pred = c_func(x)
        sse.append(np.sum((y - y_pred)**2))
    features = np.array([sse])
    # Standardize
    features_scaled = scaler.transform(features)
    # Predict
    label = model.predict(features_scaled)[0]
    return label

# Test deployment
x_new, y_new = generate_group(curves[1])  # Generate points from Curve 2 (y = x^2)
predicted_label = classify_points(x_new, y_new, model, scaler, curves)
curve_names = ['y = x', 'y = x^2', 'y = e^x']
print(f"Predicted Curve: {curve_names[predicted_label]}")

