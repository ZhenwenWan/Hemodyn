import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)

n_groups_per_curve = 100000
n_points_per_group = 10
noise_std = 0.01  # No noise

x_range = (0, 1)

def generate_group(curve_func, a, b):
    x = np.random.uniform(x_range[0], x_range[1], n_points_per_group)
    y = curve_func(x, a, b) + np.random.normal(0, noise_std, n_points_per_group)
    return x, y

def curve_A(x, a, b):
    return a * np.cos(x) + b

def curve_B(x, a, b):
    return a * x**2 + b

def curve_C(x, a, b):
    return a * np.exp(x) + b

curve_families = [curve_A, curve_B, curve_C]

X_data = []
y_data = []

for label, curve_func in enumerate(curve_families):
    for _ in range(n_groups_per_curve):
        a = np.random.uniform(0.5, 2.0)
        b = np.random.uniform(-1.0, 1.0)
        x, y = generate_group(curve_func, a, b)

        sse = []
        for c_func in curve_families:
            A_mat = np.vstack([c_func(x, 1, 0), np.ones_like(x)]).T
            coef, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
            a_fit, b_fit = coef[0], coef[1]
            y_pred = c_func(x, a_fit, b_fit)
            residuals = y - y_pred
            sse_value = np.sum(residuals ** 2)
            sse.append(sse_value)

        X_data.append(sse)
        y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model_lr.fit(X_train_scaled, y_train)

# Predict
y_pred_lr = model_lr.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"Logistic Regression with Pairwise Differences Accuracy: {accuracy_lr:.2f}")

