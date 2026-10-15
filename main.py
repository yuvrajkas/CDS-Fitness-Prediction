import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

df = pd.read_csv("cds_fitness_applicants.csv")

df["fitness_performance_score"] = (
    100
    - (df["run_1_5km_minutes"] - 5) * 12
    + df["pushups_2min"] * 0.5
    + df["situps_2min"] * 0.3
    + df["pullups"] * 1.0
    + df["flexibility_score_0_20"] * 0.5
).clip(0, 100)

features = [
    "age_years",
    "height_cm",
    "weight_kg",
    "run_1_5km_minutes",
    "pushups_2min",
    "situps_2min",
    "pullups",
    "flexibility_score_0_20"
]

X = df[features]
y_reg = df["fitness_performance_score"]
y_cls = df["meets_demo_fitness_target"]

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.25, random_state=42
)

y_cls_train = y_cls.loc[X_train.index]
y_cls_test = y_cls.loc[X_test.index]

regression_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

regression_model.fit(X_train, y_reg_train)
reg_predictions = regression_model.predict(X_test)

print("LINEAR REGRESSION RESULTS")
print("MAE:", round(mean_absolute_error(y_reg_test, reg_predictions), 2))
print("RMSE:", round(mean_squared_error(y_reg_test, reg_predictions) ** 0.5, 2))
print("R2 Score:", round(r2_score(y_reg_test, reg_predictions), 3))

plt.figure(figsize=(7, 5))
plt.scatter(y_reg_test, reg_predictions)
plt.xlabel("Actual Fitness Score")
plt.ylabel("Predicted Fitness Score")
plt.title("Linear Regression")
plt.tight_layout()
plt.savefig("regression_results.png")
plt.close()

classification_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

classification_model.fit(X_train, y_cls_train)
class_predictions = classification_model.predict(X_test)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", round(accuracy_score(y_cls_test, class_predictions), 3))
print("Precision:", round(precision_score(y_cls_test, class_predictions, zero_division=0), 3))
print("Recall:", round(recall_score(y_cls_test, class_predictions, zero_division=0), 3))
print("F1 Score:", round(f1_score(y_cls_test, class_predictions, zero_division=0), 3))

ConfusionMatrixDisplay.from_predictions(
    y_cls_test,
    class_predictions,
    labels=[0, 1],
    display_labels=["Needs Improvement", "Meets Demo Target"]
)

plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("classification_results.png")
plt.close()

results = pd.DataFrame({
    "Applicant_ID": df.loc[X_test.index, "applicant_id"].values,
    "Actual_Fitness_Score": y_reg_test.values,
    "Predicted_Fitness_Score": reg_predictions,
    "Actual_Demo_Label": y_cls_test.values,
    "Predicted_Demo_Label": class_predictions
})

results.to_csv("prediction_results.csv", index=False)
df.to_csv("cds_fitness_with_scores.csv", index=False)

print("\nFiles generated:")
print("1. regression_results.png")
print("2. classification_results.png")
print("3. prediction_results.csv")
print("4. cds_fitness_with_scores.csv")
