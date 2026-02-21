import pandas as pd
from ingestion.data_cleaner import clean_data, create_features, train_model, predict

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Example: load your raw data (replace with your actual data source)
# raw_data = pd.read_csv("your_data.csv")
# or from MT5 as a DataFrame

# Clean and create features
cleaned = clean_data(raw_data)
featured = create_features(cleaned)

# Extract features and target
X = featured[FEATURE_COLS]
y = featured["target"]

# Now X and y are ready for any model


# Example: train SVM
svm_model = SVC()
svm_model.fit(X, y)

# Example: train logistic regression
log_model = LogisticRegression()
log_model.fit(X, y)

# For regression tasks (if you want to predict returns instead of direction)
# you can use the 'return' column as target, but note that create_features
# already computed 'target' as binary. Adjust as needed.


# ATR filter

data['ATR_mean'] = data['ATR'].rolling(50).mean()

data.loc[data['ATR'] <= data ['ATR_mean'], 'signal'] = 0


    # CREATE SL TP
data['SL'] = None
data['TP'] = None

data.loc[data['signal'] == 1, 'SL'] = data['close'] - data['ATR'] * 2
data.loc[data['signal'] == 1, 'TP'] = data['close'] + data['ATR'] * 4


data.loc[data['signal'] == -1, 'SL'] = data['close'] + data['ATR'] * 2
data.loc[data['signal'] == -1, 'TP'] = data['close'] - data['ATR'] * 4

print("Signals with SL TP generated")


generate_signals()
