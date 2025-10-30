import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

class TemporalLoadClassifier:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.results = {}
        self.feature_importance = None

    def create_temporal_features(self):
        """Create time-series features that actually predict load"""
        # Sort by time to ensure proper lags
        self.df = self.df.sort_values(['YEAR', 'Month', 'Day', 'Hour']).reset_index(drop=True)

        # Create target - 3 balanced classes
        load = self.df['Electric Load (MW)']
        terciles = load.quantile([0.33, 0.66]).tolist()
        self.df['Load_Level'] = pd.cut(load, bins=[load.min(), terciles[0], terciles[1], load.max()],
                                     labels=['Low', 'Medium', 'High'], include_lowest=True)

        print("Class distribution:")
        print(self.df['Load_Level'].value_counts().sort_index())

        # CRITICAL: Lag features (past load values)
        self.df['Load_Lag_1'] = self.df['Electric Load (MW)'].shift(1)  # Previous hour
        self.df['Load_Lag_2'] = self.df['Electric Load (MW)'].shift(2)  # 2 hours ago
        self.df['Load_Lag_3'] = self.df['Electric Load (MW)'].shift(3)  # 3 hours ago
        self.df['Load_Lag_24'] = self.df['Electric Load (MW)'].shift(24)  # Same time yesterday

        # Rolling statistics
        self.df['Load_Rolling_Mean_3'] = self.df['Electric Load (MW)'].rolling(3, min_periods=1).mean()
        self.df['Load_Rolling_Std_3'] = self.df['Electric Load (MW)'].rolling(3, min_periods=1).std()

        # Time of day patterns (most important!)
        self.df['Hour_Sin'] = np.sin(2 * np.pi * self.df['Hour'] / 24)
        self.df['Hour_Cos'] = np.cos(2 * np.pi * self.df['Hour'] / 24)

        # Day of week patterns
        self.df['Day_of_Week'] = self.df['Day'] % 7
        self.df['Is_Weekend'] = ((self.df['Day_of_Week'] == 0) | (self.df['Day_of_Week'] == 6)).astype(int)

        # Peak hours
        self.df['Is_Peak_Hour'] = ((self.df['Hour'] >= 17) & (self.df['Hour'] <= 21)).astype(int)
        self.df['Is_Night'] = ((self.df['Hour'] < 6) | (self.df['Hour'] > 22)).astype(int)

        # Weather features that actually affect load
        self.df['Heating_Degrees'] = np.maximum(18 - self.df['temperature'], 0)  # Heating needed
        self.df['Cooling_Degrees'] = np.maximum(self.df['temperature'] - 24, 0)  # Cooling needed
        self.df['Comfort_Index'] = self.df['temperature'] - (0.55 * (1 - self.df['specific humidity']/100) * (self.df['temperature'] - 58))

        # Fill NaN values from lag features
        self.df = self.df.fillna(method='bfill')

        # Select the most predictive features
        features = [
            # PAST LOAD PATTERNS (most important!)
            'Load_Lag_1', 'Load_Lag_2', 'Load_Lag_3', 'Load_Lag_24',
            'Load_Rolling_Mean_3', 'Load_Rolling_Std_3',

            # TIME PATTERNS
            'Hour', 'Hour_Sin', 'Hour_Cos', 'Day_of_Week', 'Is_Weekend',
            'Is_Peak_Hour', 'Is_Night', 'Month',

            # WEATHER PATTERNS
            'temperature', 'specific humidity', 'wind speed', 'irradiance',
            'Heating_Degrees', 'Cooling_Degrees', 'Comfort_Index'
        ]

        X = self.df[features]
        y = LabelEncoder().fit_transform(self.df['Load_Level'])

        return X, y

    def plot_feature_importance(self, feature_importance, top_n=15):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(top_n)

        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)

        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, results, y_test):
        """Plot confusion matrices for all models"""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

        if n_models == 1:
            axes = [axes]

        class_names = ['Low', 'Medium', 'High']

        for idx, (model_name, result) in enumerate(results.items()):
            y_pred = result['predictions']
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, results):
        """Plot model performance comparison"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        model_names = list(results.keys())

        # Calculate metrics for each model
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1']
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for idx, metric in enumerate(metrics):
            bars = axes[idx].bar(comparison_df['Model'], comparison_df[metric],
                               color=colors[:len(model_names)], alpha=0.8)
            axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
            axes[idx].set_ylabel(metric)
            axes[idx].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

            axes[idx].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

        return comparison_df

    def plot_roc_curves(self, results, X_test_scaled, y_test):
        """Plot ROC curves for all models and classes"""
        plt.figure(figsize=(10, 8))

        # Colors for different classes
        colors = cycle(['#2E86AB', '#A23B72', '#F18F01'])
        class_names = ['Low', 'Medium', 'High']

        for model_name, result in results.items():
            model = result['model']

            # Get predicted probabilities for each class
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test_scaled)
            else:
                # For models without predict_proba, use decision function
                y_score = model.decision_function(X_test_scaled)
                # Convert to probabilities using softmax
                y_score = np.exp(y_score) / np.exp(y_score).sum(axis=1, keepdims=True)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(len(class_names)):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot ROC curves for each class
            for i, color in zip(range(len(class_names)), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{model_name} - {class_names[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run_temporal_analysis(self):
        """Run analysis with temporal features"""
        print("TEMPORAL LOAD CLASSIFICATION")
        print("=" * 50)

        X, y = self.create_temporal_features()

        print(f"Class distribution: {np.bincount(y)}")
        print(f"Features used: {len(X.columns)}")
        print(f"Sample size: {len(X)}")

        # Remove any remaining NaN values
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Optimized models
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        }

        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Calculate all metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class,
                'predictions': y_pred,
                'model': model,
                'true_labels': y_test
            }

            print(f"  {name} Accuracy: {accuracy:.4f}")
            print(f"  {name} Precision: {precision:.4f}")
            print(f"  {name} Recall: {recall:.4f}")
            print(f"  {name} F1-Score: {f1:.4f}")

        # Store results
        self.results = results

        # Feature importance
        print(f"\nTop 10 Feature Importance:")
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': results['XGBoost']['model'].feature_importances_
        }).sort_values('importance', ascending=False)

        print(self.feature_importance.head(10).to_string(index=False))

        # Plot feature importance
        self.plot_feature_importance(self.feature_importance)

        # Plot confusion matrices
        self.plot_confusion_matrices(results, y_test)

        # Plot model comparison
        comparison_df = self.plot_model_comparison(results)

        # Plot ROC curves
        self.plot_roc_curves(results, X_test_scaled, y_test)

        # Detailed report
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\nDetailed report for {best_model_name}:")
        print(classification_report(y_test, results[best_model_name]['predictions'],
                                 target_names=['Low', 'Medium', 'High']))

        # Print per-class metrics for best model
        best_model_results = results[best_model_name]
        print(f"\nPer-class metrics for {best_model_name}:")
        class_metrics_df = pd.DataFrame({
            'Class': ['Low', 'Medium', 'High'],
            'Precision': best_model_results['precision_per_class'],
            'Recall': best_model_results['recall_per_class'],
            'F1-Score': best_model_results['f1_per_class']
        })
        print(class_metrics_df.to_string(index=False))

        best_acc = max([result['accuracy'] for result in results.values()])
        print(f"\nBEST ACCURACY: {best_acc:.4f}")

        return results, self.feature_importance

# Try binary classification with temporal features
def binary_temporal_classification(data_path):
    """Binary classification with temporal features"""
    print("\nBINARY TEMPORAL CLASSIFICATION")
    print("=" * 50)

    df = pd.read_csv(data_path)

    # Sort by time
    df = df.sort_values(['YEAR', 'Month', 'Day', 'Hour']).reset_index(drop=True)

    # Binary target
    median_load = df['Electric Load (MW)'].median()
    df['Is_High_Load'] = (df['Electric Load (MW)'] > median_load).astype(int)

    print("Binary class distribution:")
    print(df['Is_High_Load'].value_counts().sort_index())

    # Temporal features
    df['Load_Lag_1'] = df['Electric Load (MW)'].shift(1)
    df['Load_Lag_24'] = df['Electric Load (MW)'].shift(24)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Is_Peak_Hour'] = ((df['Hour'] >= 17) & (df['Hour'] <= 21)).astype(int)

    df = df.fillna(method='bfill')

    features = [
        'Load_Lag_1', 'Load_Lag_24', 'Hour_Sin', 'Hour_Cos', 'Is_Peak_Hour',
        'temperature', 'specific humidity', 'wind speed', 'irradiance', 'Hour'
    ]

    X = df[features]
    y = df['Is_High_Load']

    # Remove NaN
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    print(f"Binary Temporal Accuracy: {accuracy:.4f}")
    print(f"Binary Temporal Precision: {precision:.4f}")
    print(f"Binary Temporal Recall: {recall:.4f}")
    print(f"Binary Temporal F1-Score: {f1:.4f}")

    # Plot binary confusion matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.title('Binary Classification\nConfusion Matrix', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # Plot ROC curve for binary classification
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Binary Classification ROC Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Binary Classification AUC Score: {roc_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc

# Run the analyses
if __name__ == "__main__":
    # Approach 1: Temporal 3-class
    print("APPROACH 1: TEMPORAL 3-CLASS")
    classifier = TemporalLoadClassifier('weather and load dataset.csv')
    results, importance = classifier.run_temporal_analysis()

    print("\n" + "="*60)

    # Approach 2: Binary temporal
    print("APPROACH 2: BINARY TEMPORAL")
    binary_acc, binary_precision, binary_recall, binary_f1, binary_auc = binary_temporal_classification('weather and load dataset.csv')

    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    # Get best 3-class model results
    best_3class_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_3class_acc = results[best_3class_name]['accuracy']
    best_3class_precision = results[best_3class_name]['precision']
    best_3class_recall = results[best_3class_name]['recall']
    best_3class_f1 = results[best_3class_name]['f1']

    comparison_data = {
        'Approach': ['3-Class Temporal', 'Binary Temporal'],
        'Best Model': [best_3class_name, 'XGBoost'],
        'Accuracy': [best_3class_acc, binary_acc],
        'Precision': [best_3class_precision, binary_precision],
        'Recall': [best_3class_recall, binary_recall],
        'F1-Score': [best_3class_f1, binary_f1],
        'AUC Score': ['Multi-class', f'{binary_auc:.4f}']
    }

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    best_overall = max(best_3class_acc, binary_acc)
    print(f"\n FINAL BEST ACCURACY: {best_overall:.4f}")
