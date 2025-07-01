"""
Ta datoteka naj ob zagonu (brez dodatnih parametrov) v 5 minutah ustvari vašo končno oddajo.
"""

"""

type_of_target = 'multiclass-multioutput'
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error


train = pd.read_csv("../podatki/bicikelj_train.csv")
test = pd.read_csv("../podatki/bicikelj_test.csv")


train['timestamp'] = pd.to_datetime(train['timestamp'])

idx = 0
# Loop over rows (timestamps)
for index, row in train.iterrows():
    timestamp = row['timestamp']
    
    # Loop over each station column
    for station in train.columns[1:]:  # skip 'timestamp'
        
        print(f"At {timestamp}, station '{station}' had {row[station]} bikes.")

        idx += 1
        if idx > 5:
            break

    if idx > 5:
        break


# Load data
df = pd.read_csv("../podatki/bicikelj_train.csv")

# Convert timestamp and sort
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

station_columns = df.columns.difference(['timestamp', 'hour', 'dayofweek'])
df = df.dropna(subset=station_columns)


# Features: only time info
X = df[['hour', 'dayofweek']]

print(X[:5])

# Targets: all stations (exclude timestamp and time features)
Y = df.drop(columns=['timestamp', 'hour', 'dayofweek'])

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Wrap linear regression with MultiOutputRegressor
model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, Y_train)

# Predict all stations
Y_pred = model.predict(X_test)

# Mean absolute error across all stations and all timestamps
overall_mae = mean_absolute_error(Y_test, Y_pred)
print(f"Overall MAE: {overall_mae:.2f}")

"""
# MAE per station
mae_per_station = (abs(Y_test.values - Y_pred)).mean(axis=0)
for station, mae in zip(Y.columns, mae_per_station):
    print(f"{station}: MAE = {mae:.2f}")
"""



# === Final evaluation on bicikelj_test.csv ===
test_df = pd.read_csv("../podatki/bicikelj_test.csv")
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
test_df = test_df.sort_values('timestamp')

# Extract features
test_df['hour'] = test_df['timestamp'].dt.hour
test_df['dayofweek'] = test_df['timestamp'].dt.dayofweek

# Make sure there are no missing values
test_df = test_df.dropna(subset=station_columns)

# Prepare test features and labels
X_final = test_df[['hour', 'dayofweek']]
Y_final = test_df.drop(columns=['timestamp', 'hour', 'dayofweek'])

# Predict and evaluate
Y_final_pred = model.predict(X_final)
final_mae = mean_absolute_error(Y_final, Y_final_pred)
print(f"\nFINAL evaluation on bicikelj_test.csv - MAE: {final_mae:.2f}")

"""
final_mae_per_station = (abs(Y_final.values - Y_final_pred)).mean(axis=0)
for station, mae in zip(Y.columns, final_mae_per_station):
    print(f"{station}: MAE = {mae:.2f}")
"""
predictions_df = pd.DataFrame(Y_final_pred, columns=Y_final.columns)
predictions_df.insert(0, 'timestamp', test_df['timestamp'])
predictions_df.to_csv("sklearn_linreg_pred.csv", index=False)

