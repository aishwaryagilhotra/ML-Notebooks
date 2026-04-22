# ML-Notebooks
A curated portfolio of my machine learning experiments and mini-projects. Includes implementations, visualizations, and insights across supervised and unsupervised learning models.

# Basics
- How to Split a Dataset into train, test, and validation sets - DatasetSplit.py
- Linear Regression Positive and negative slopes - LinearRegression.py
- How to Gradient Descent - GradientDescent.py
- How to Naive Bayes - NaiveBayes.py
- Evaluating Regression models - EvaluationMetrices.py

# Mini Projects-
1. Predict Energy Demand during Extreme Weather - XGBoost, Random Forest
2. Bank Notes Authentication - Decision Tree
3. AI-Powered Financial Advisor - Random Forest, XGBoost, LightGBM (Both Regression and Classification)
4. A comprehensive demand forecasting system that predicts future product demand (Low/Medium/High) using historical sales data, seasonal patterns, competitor pricing, marketing campaigns, and customer reviews - Random Forest, Decision Tree, XGBoost, LightGBM

# Satellite Data

1. Random_Forest_Sentinel_2 -
   - Matches satellite tiles with masks using filename suffix
   - Loads data, reshapes pixels into feature vectors (7 bands) and labels
   - Samples 2% of pixels to reduce size
   - Combines into dataset (X, y) and checks stats
   - Trains a Random Forest classifier for pixel-wise classification
   - Evaluates performance with classification report
   - Shows feature importance (bands + indices)
   - Visualizes RGB image, ground truth mask, and predicted mask

# Datasets-
1. Predict Energy Demand during Extreme Weather - https://www.kaggle.com/datasets/orvile/weather-and-electric-load-dataset
2. Bank Notes Authentication - https://www.kaggle.com/datasets/catiely05/data-banknote-authentication/code
3. AI-Powered Financial Advisor - Consists of 3 datasets
   3.1 Bank Interest Rates - https://www.kaggle.com/datasets/jaskiratsinghjassi/indian-banks-interest-rates
   3.2 Monetary Policies - RBI DBIE https://data.rbi.org.in/DBIE/
   3.3 Synthetic Customer Dataset - GPT-5
4. DatasetSplit - Calls OpenML API for recent data(first 500 rows) on tasks and algorithms used
5. LinearRegression + Gradient Descent - https://archive.ics.uci.edu/dataset/9/auto+mp
6. Demand Forecasting - Synthetic (python pandas)
7. Random Forest Setinel2 - BHOONIDHI https://bhoonidhi.nrsc.gov.in/bhoonidhi/home.html#services
