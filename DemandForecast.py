import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("="*70)
print("DEMAND FORECASTING")
print("="*70)

# ============================================
# STEP 1: CREATE DATA WITH EXACT REQUIRED COLUMNS
# ============================================

print("\n[1] Creating dataset with exact required columns...")

np.random.seed(42)
n_days = 1095  # 3 years
dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

# Initialize arrays for the EXACT columns we need
past_sales = []
seasonal_trends = []  # This will be a composite score
competitor_pricing = []
marketing_campaign = []  # This will be spend amount
customer_reviews = []

for i, date in enumerate(dates):
    # ========== PAST SALES (with pattern) ==========
    # Base sales with trend and noise
    if i == 0:
        prev_sales = 100
    else:
        prev_sales = past_sales[i-1]
    
    # Add some pattern to sales
    month = date.month
    day_of_week = date.dayofweek
    
    # Weekend effect
    if day_of_week >= 5:
        weekend_boost = 30
    else:
        weekend_boost = 10
    
    # Month effect
    if month in [11, 12]:  # Holiday season
        month_boost = 60
    elif month in [6, 7, 8]:  # Summer
        month_boost = 30
    else:
        month_boost = 15
    
    # Marketing impact on sales
    # Black Friday / Cyber Monday
    if (month == 11 and date.day >= 20) or (month == 12 and date.day <= 5):
        marketing_impact = 120
        marketing_spend = 20000
    # Christmas
    elif month == 12 and date.day >= 15:
        marketing_impact = 100
        marketing_spend = 15000
    # Monthly campaigns
    elif date.day <= 7:
        marketing_impact = 40
        marketing_spend = 8000
    else:
        marketing_impact = np.random.uniform(0, 10)
        marketing_spend = np.random.uniform(0, 1000)
    
    # Calculate current sales based on previous sales + patterns
    noise = np.random.normal(0, 8)
    current_sales = prev_sales * 0.7 + weekend_boost + month_boost + marketing_impact + noise
    current_sales = max(current_sales, 20)  # Ensure positive
    
    past_sales.append(current_sales)
    
    # ========== SEASONAL TRENDS (composite score 0-100) ==========
    # Combine multiple seasonal factors into one score
    # Month seasonality (0-40)
    if month in [12, 1]:  # Winter holidays
        month_score = 40
    elif month in [11, 2]:  # Shoulder months
        month_score = 30
    elif month in [6, 7, 8]:  # Summer
        month_score = 35
    else:
        month_score = 20
    
    # Week seasonality (0-30)
    week_of_month = (date.day - 1) // 7 + 1
    if week_of_month == 1:  # First week - payday effect
        week_score = 30
    elif week_of_month == 4:  # Last week
        week_score = 25
    else:
        week_score = 20
    
    # Day of week seasonality (0-30)
    if day_of_week >= 5:  # Weekend
        day_score = 30
    else:
        day_score = 15
    
    # Composite seasonal trend score (0-100)
    seasonal_score = (month_score * 0.4 + week_score * 0.3 + day_score * 0.3)
    seasonal_trends.append(seasonal_score)
    
    # ========== COMPETITOR PRICING ==========
    # Our price and competitor price with relationship
    if marketing_spend > 5000:
        our_price = 45 + np.random.normal(0, 1.5)
        comp_price = 52 + np.random.normal(0, 2)
    else:
        our_price = 55 + np.random.normal(0, 1.5)
        comp_price = 50 + np.random.normal(0, 2)
    
    # Store the price difference as competitor pricing metric
    # Positive means our price is higher than competitor
    price_diff = our_price - comp_price
    competitor_pricing.append(price_diff)
    
    # ========== MARKETING CAMPAIGN ==========
    # Already calculated above as marketing_spend
    marketing_campaign.append(marketing_spend)
    
    # ========== CUSTOMER REVIEWS ==========
    # Reviews based on marketing and price
    if marketing_spend > 5000:
        rating = 4.6 + np.random.normal(0, 0.2)
    elif our_price < 48:
        rating = 4.3 + np.random.normal(0, 0.3)
    else:
        rating = 3.9 + np.random.normal(0, 0.4)
    
    rating = np.clip(rating, 1, 5)
    customer_reviews.append(rating)

# Create DataFrame with EXACT required columns
df = pd.DataFrame({
    'date': dates,
    'past_sales': past_sales,
    'seasonal_trends': seasonal_trends,
    'competitor_pricing': competitor_pricing,
    'marketing_campaign': marketing_campaign,
    'customer_reviews': customer_reviews
})

print(f"Dataset created: {len(df)} rows")
print(f"\nColumns: {list(df.columns)}")

# ============================================
# STEP 2: FEATURE ENGINEERING (WITHOUT CREATING NEW COLUMNS)
# ============================================

print("\n[2] Engineering features from existing columns...")

def engineer_features(df):
    df = df.copy()
    
    # ========== FROM PAST SALES ==========
    # Lag features (more past sales)
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'past_sales_lag_{lag}'] = df['past_sales'].shift(lag)
    
    # Rolling statistics from past sales
    for window in [7, 14, 30]:
        df[f'past_sales_mean_{window}'] = df['past_sales'].shift(1).rolling(window, min_periods=1).mean()
        df[f'past_sales_std_{window}'] = df['past_sales'].shift(1).rolling(window, min_periods=1).std()
    
    # ========== FROM SEASONAL TRENDS ==========
    # Decompose seasonal trends into components
    df['trend_lag_1'] = df['seasonal_trends'].shift(1)
    df['trend_lag_7'] = df['seasonal_trends'].shift(7)
    df['trend_change'] = df['seasonal_trends'].diff()
    
    # ========== FROM COMPETITOR PRICING ==========
    df['pricing_lag_1'] = df['competitor_pricing'].shift(1)
    df['pricing_lag_7'] = df['competitor_pricing'].shift(7)
    df['pricing_rolling_mean_7'] = df['competitor_pricing'].shift(1).rolling(7, min_periods=1).mean()
    df['is_price_advantage'] = (df['competitor_pricing'] < 0).astype(int)  # Negative means our price is lower
    
    # ========== FROM MARKETING CAMPAIGN ==========
    df['marketing_lag_1'] = df['marketing_campaign'].shift(1)
    df['marketing_lag_7'] = df['marketing_campaign'].shift(7)
    df['marketing_rolling_mean_7'] = df['marketing_campaign'].shift(1).rolling(7, min_periods=1).mean()
    df['marketing_rolling_sum_30'] = df['marketing_campaign'].shift(1).rolling(30, min_periods=1).sum()
    df['has_marketing'] = (df['marketing_campaign'] > 100).astype(int)
    
    # ========== FROM CUSTOMER REVIEWS ==========
    df['reviews_lag_1'] = df['customer_reviews'].shift(1)
    df['reviews_lag_7'] = df['customer_reviews'].shift(7)
    df['reviews_rolling_mean_7'] = df['customer_reviews'].shift(1).rolling(7, min_periods=1).mean()
    df['is_high_rating'] = (df['customer_reviews'] >= 4).astype(int)
    
    # ========== INTERACTION FEATURES (between original columns) ==========
    df['marketing_x_reviews'] = df['marketing_campaign'] * df['customer_reviews']
    df['price_x_marketing'] = df['competitor_pricing'] * df['marketing_campaign']
    df['trend_x_marketing'] = df['seasonal_trends'] * df['marketing_campaign']
    df['sales_x_marketing'] = df['past_sales'] * df['marketing_campaign']
    
    return df

# Apply feature engineering
df_featured = engineer_features(df)

# Drop NaN values
df_featured = df_featured.dropna().reset_index(drop=True)

print(f"Total features after engineering: {df_featured.shape[1]}")
print(f"Final shape: {df_featured.shape}")

# ============================================
# STEP 3: CREATE DEMAND CATEGORIES (based on past_sales)
# ============================================

print("\n[3] Creating demand categories...")

# Create 3 categories based on past_sales
p33 = df_featured['past_sales'].quantile(0.33)
p67 = df_featured['past_sales'].quantile(0.67)

df_featured['demand'] = pd.cut(
    df_featured['past_sales'],
    bins=[-np.inf, p33, p67, np.inf],
    labels=['Low', 'Medium', 'High']
)

print("Class distribution:")
print(df_featured['demand'].value_counts())
print("\nPercentage distribution:")
print(df_featured['demand'].value_counts(normalize=True).round(3) * 100)

# ============================================
# STEP 4: SAVE TO CSV (WITH ALL ENGINEERED FEATURES)
# ============================================

print("\n[4] Saving to CSV...")

df_featured.to_csv('demand_dataset_complete.csv', index=False)
print("✓ Saved: demand_dataset_complete.csv")

# Save original columns only for clarity
df[['date', 'past_sales', 'seasonal_trends', 'competitor_pricing', 
    'marketing_campaign', 'customer_reviews']].to_csv('demand_dataset_original.csv', index=False)
print("✓ Saved: demand_dataset_original.csv")

# ============================================
# STEP 5: PREPARE FOR MODELING
# ============================================

print("\n[5] Preparing for modeling...")

# Features (exclude date, past_sales, demand)
exclude_cols = ['date', 'past_sales', 'demand']
feature_cols = [col for col in df_featured.columns if col not in exclude_cols]

X = df_featured[feature_cols]
y = df_featured['demand']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split (80-20, maintaining time order)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y_encoded[:train_size], y_encoded[train_size:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {len(feature_cols)}")

# ============================================
# STEP 6: TRAIN MODELS 
# ============================================

print("\n[6] Training classification models...")
print("-" * 50)

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
}

results = []

for name, model in models.items():
    print(f"\n► {name}")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1_Score': round(f1, 4)
    })
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

# ============================================
# STEP 7: DISPLAY RESULTS
# ============================================

print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('model_results.csv', index=False)
print("\n✓ Results saved to 'model_results.csv'")

# ============================================
# STEP 8: BEST MODEL DETAILS
# ============================================

best_model_idx = results_df['F1_Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_f1 = results_df.loc[best_model_idx, 'F1_Score']

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model_name} (F1 Score: {best_f1})")
print('='*70)

# Get best model object
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("\nConfusion Matrix:")
print(cm_df)

