import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Load datasets with correct file paths
advisor_df = pd.read_csv('/content/ai_financial_advisor_dataset.csv')
banks_df = pd.read_csv('/content/Banks-Interest-Rates.csv')
monetary_df = pd.read_csv('/content/Copy_of_monetary_policies.csv')

# Preprocessing
advisor_df.fillna(method='ffill', inplace=True)
label_encoder = LabelEncoder()
advisor_df['Advice_Summary'] = label_encoder.fit_transform(advisor_df['Advice_Summary'])

# Features and targets
X = advisor_df[['Age', 'Income']]
y_class = advisor_df['Advice_Summary']
y_regr = advisor_df[['Stocks_%', 'Bonds_%', 'FD_%']]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_regr, X_test_regr, y_train_regr, y_test_regr = train_test_split(X, y_regr, test_size=0.2, random_state=42)

# Initialize models
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

regressors = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42)
}

# Train classifiers and visualize results
clf_results = {}
for name, model in classifiers.items():
    model.fit(X_train_clf, y_train_clf)
    y_pred = model.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred)
    print(f'\n=== {name} Classifier ===')
    print(f'Accuracy: {acc:.4f}')
    print('Classification Report:')
    print(classification_report(y_test_clf, y_pred, target_names=label_encoder.classes_))

    clf_results[name] = acc

    # Confusion Matrix
    cm = confusion_matrix(y_test_clf, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

    # Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(8,6))
        plt.barh(X.columns, model.feature_importances_)
        plt.xlabel('Feature Importance')
        plt.title(f'{name} Feature Importance')
        plt.show()

# Train regressors and visualize results per target
for target in ['Stocks_%', 'Bonds_%', 'FD_%']:
    print(f'\n▶ Regression Results for Target: {target}')
    regr_results = {}

    for name, model in regressors.items():
        model.fit(X_train_regr, y_train_regr[target])
        y_pred_regr = model.predict(X_test_regr)
        rmse = np.sqrt(mean_squared_error(y_test_regr[target], y_pred_regr))
        r2 = r2_score(y_test_regr[target], y_pred_regr)

        print(f'{name} – RMSE: {rmse:.4f}, R²: {r2:.4f}')
        regr_results[name] = {'RMSE': rmse, 'R2': r2}

        # Residual Plot
        residuals = y_test_regr[target] - y_pred_regr
        plt.figure(figsize=(8,6))
        plt.scatter(y_pred_regr, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel(f'Predicted {target}')
        plt.ylabel('Residuals')
        plt.title(f'{name} Residual Plot ({target})')
        plt.show()

        # Predicted vs Actual Scatter Plot
        plt.figure(figsize=(8,6))
        plt.scatter(y_test_regr[target], y_pred_regr, alpha=0.7)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'{name} Predicted vs Actual ({target})')
        plt.show()

    # Bar Chart Comparison for This Target
    plt.figure(figsize=(10,6))
    rmse_values = [metrics['RMSE'] for metrics in regr_results.values()]
    r2_values = [metrics['R2'] for metrics in regr_results.values()]
    x = list(regr_results.keys())
    width = 0.35
    plt.bar(x, rmse_values, width=width, label='RMSE')
    plt.bar(x, r2_values, width=width, label='R² Score', alpha=0.7)
    plt.xlabel('Regressor')
    plt.ylabel('Score')
    plt.title(f'Regressor Performance Comparison – {target}')
    plt.legend()
    plt.show()

# Overall Classifier Accuracy Comparison
plt.figure(figsize=(10,6))
plt.bar(clf_results.keys(), clf_results.values())
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.show()
