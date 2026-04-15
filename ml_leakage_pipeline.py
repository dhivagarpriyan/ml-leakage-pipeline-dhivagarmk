#Task 1 — Reproduce and Identify Leakage
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# WRONG: Scaling entire dataset before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split AFTER scaling (causes leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy scores
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

#Task 2 — Fix Workflow Using Pipeline + Cross-Validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Cross-validation
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))
print("Standard deviation:", np.std(cv_scores))

#Task 3 — Decision Tree Depth Experiment
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

depth_values = [1, 5, 20]

results = []

for depth in depth_values:
    
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(
        y_train,
        model.predict(X_train)
    )
    
    test_acc = accuracy_score(
        y_test,
        model.predict(X_test)
    )
    
    results.append((depth, train_acc, test_acc))

# Display results
print("Depth | Train Accuracy | Test Accuracy")
for r in results:
    print(r)
