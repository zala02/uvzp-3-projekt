# basic libraries
import pandas as pd
import numpy as np

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

# additional information about holidays
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from podatki.holidays import add_holiday_features



TRAIN_SIZE = 48
PREDICT_SIZE = 4
GAP_SIZE = 10
BATCH_SIZE = 32  # velikost podatkov pri paketnem učenju
EPOCHS = 30  # število iteracij učenja
LEARNING_RATE = 0.005  # stopnja učenja

# ToTensorDataset modificiran iz 020-nn.py
# mormo mal spremenit, ker nimamo samo dveh moznih outputov 
# ustvarimo razred, ki pretvorti numpy v tenzor
class ToTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# LoadData modificiran iz 020-nn.py 
# pomozni razred za nalaganje in pripravo podatkov za ucenje in testiranje modelov v pytorch okolju
class PrepareData:
    def __init__(self, filename, train_size=48, predict_size=4, gap_size=10, batch_size=32):

        print("***\nInitialization of PrepareData started")

        # store the variables
        self.filename = filename
        self.train_size = train_size
        self.predict_size = predict_size
        self.gap_size = gap_size
        self.batch_size = batch_size
        
        # store the original data since im going to modify it (just in case, can delete that later)
        self.og_data = pd.read_csv(filename)

        # other variables
        self.station_columns = None
        self.attributes = []
        
        # first we load data, possibly process it, add attributes if needed
        self.data = self._load_and_process()

        self.X_train, self.X_val, self.y_train, self.y_val, self.timestamps_train, self.timestamps_val = self._create_train_test()

        print("***\nSuccessfully initialized class PrepareData")

    def _load_and_process(self):
        """Load data, drop nan values, add atributes."""

        print("***\nLoading and processing data started")

        data = pd.read_csv(self.filename)

        # transform date
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp')

        # Add holiday flags
        data = add_holiday_features(data)  # must return is_holiday and is_school_holiday

        # Extract time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['month'] = data['timestamp'].dt.month
        data['is_weekend'] = data['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)

        # One-hot encode hour and month
        hour_ohe = pd.get_dummies(data['hour'], prefix='hour')
        month_ohe = pd.get_dummies(data['month'], prefix='month')
        data = pd.concat([data.drop(columns=['hour', 'month']), hour_ohe, month_ohe], axis=1)

        # Set attribute and target columns
        self.attributes = list(hour_ohe.columns) + list(month_ohe.columns) + ['is_weekend', 'is_holiday', 'is_school_holiday']
        self.station_columns = data.columns.difference(['timestamp'] + self.attributes)

        # Create safe lag features
        for lag in [1, 2]:
            lagged_df = data[['timestamp'] + list(self.station_columns)].copy()
            lagged_df['timestamp'] = lagged_df['timestamp'] + pd.Timedelta(hours=lag)
            lagged_df = lagged_df.rename(columns={col: f"{col}_lag{lag}" for col in self.station_columns})
            data = data.merge(lagged_df, on='timestamp', how='left')
        
        before_drop = data.shape[0]
        # Drop rows where any of the targets or lagged inputs are missing
        required_columns = list(self.station_columns) + [f"{col}_lag{lag}" for col in self.station_columns for lag in [1, 2]]
        data = data.dropna(subset=required_columns).reset_index(drop=True)
        after_drop = data.shape[0]
        dropped = before_drop - after_drop
        print(f"Dropped {dropped} rows due to missing lag values.")

        # Add lag features to attribute list
        lag_attributes = [f"{col}_lag{lag}" for col in self.station_columns for lag in [1, 2]]
        self.attributes += lag_attributes

        print(f"Loading and processing data finished. For now we have following attributes: {self.attributes}")
        print(f"Now we have {len(self.attributes)} attributes")

        return data

    def _create_train_test(self):
        """Divide data to <train_size> and <predict_size>, drop <gap_size>."""

        
        print("***\nCreating train and test sets started")


        # Extract features (X) and targets (y)
        X = self.data[self.attributes].values.astype(np.float32)  # shape: (N, num_features)
        y = self.data[self.station_columns].values.astype(np.float32)  # shape: (N, num_stations)
        timestamps = self.data['timestamp'].values  # for optional inspection or plotting

        # Drop rows with any NaNs in X or y
        valid_rows = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
        X = X[valid_rows]
        y = y[valid_rows]
        timestamps = timestamps[valid_rows]

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        timestamps_train, timestamps_val = timestamps[:split_idx], timestamps[split_idx:]

        """
        # Print example training and validation samples
        print("\n--- Sample X_train ---")
        print(X_train[0])  # shape: (48, 2)
        print("\n--- Sample y_train ---")
        print(y_train[0])  # shape: (4, num_stations)

        print("\n--- Sample X_val ---")
        print(X_val[0])  # shape: (48, 2)
        print("\n--- Sample y_val ---")
        print(y_val[0])  # shape: (4, num_stations)

        print("\n--- Sample timestamps_train ---")
        print(timestamps_train[0])  # shape: (4,)

        print("\n--- Sample timestamps_val ---")
        print(timestamps_val[0])  # shape: (4,)
        """
       
        print("Creating train and test sets finished")

        return X_train, X_val, y_train, y_val, timestamps_train, timestamps_val


    def get_train_loader(self):
        """Creates and returns a PyTorch data object for training."""

        # ce zelimo podatke v paketih, poklicemo to funkcijo
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        return X_train_tensor, y_train_tensor


    def get_test_data(self):
        """Returns validation data as PyTorch tensors."""
        X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32)
        return X_val_tensor, y_val_tensor






class Model_Neural_Network(nn.Module):

    # input_len : how many hours are we using for training (48h in our case)
    # input_dim : number of features per time step (in our case ['hour', 'is_weekend', 'is_holiday', 'is_school_holiday'] so 4)
    # output_len: number of hours we want to predict (in our case for 4 hours)
    # num_stations : number of bike stations we're making predicitions for

    def __init__(self, input_dim=207, num_stations=84, lr=0.001, epochs=1500, lambda_reg=0.00001):
        super().__init__()

        self.input_size = input_dim
        self.output_size = num_stations
        self.num_stations = num_stations

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            #nn.Dropout(0.2),  # drop 20% neurons

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            #nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.Dropout(0.2),  # drop 20% neurons

            nn.Linear(128, self.output_size)
        )

        self.lr = lr 
        self.lambda_reg = lambda_reg
        self.epochs = epochs

        self.scaler = StandardScaler()


    def forward(self, x):
        return self.model(x)  # x shape: [batch_size, input_dim]


    def soft_threshold(self, param, lmbd):
        with torch.no_grad():
            # postavi na 0 utezi, ki so manjse od lmbd
            # tiste, ko so vecje, pa zmanjsa za lmbd
            # param.copy_() je funkcija, ki lahko neposredno popravi utezi
            param.copy_(param.sign() * torch.clamp(param.abs() - lmbd, min=0.0))


    def fit(self, X_train, y_train):
        print("************\nTRAINING")
        # 1. Fit the scaler on all training data
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)

        # 2. Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        #print(f"x tensor shape : {X_tensor.shape}")
        #print(f"y tensor shape : {y_tensor.shape}")

        #print(f"loss function: {self.lr}")
        #print(f"parameters: {self.parameters()}")
        # 3. Define optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        
        #print("y_train contains NaN:", torch.isnan(y_train).any().item())
        #print("y_train contains Inf:", torch.isinf(y_train).any().item())


        # 4. Training loop
        for epoch in range(self.epochs):
            self.train()

            pred = self(X_tensor)

            mse_loss = loss_fn(pred, y_tensor)

            # Add L1 regularization
            l1_norm = sum(param.abs().sum() for name, param in self.named_parameters() if 'weight' in name)
            loss = mse_loss + self.lambda_reg * l1_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Apply soft thresholding (sparsity)
            for name, param in self.named_parameters():
                if 'weight' in name:
                    self.soft_threshold(param, self.lambda_reg)

            if epoch % 200 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    def predict(self, X):
 
        self.eval()
        with torch.no_grad():
            
            # standardize data
            X_np = X.numpy()
            X_normed = self.scaler.transform(X_np)
            X_tensor = torch.tensor(X_normed, dtype=torch.float32)
            
            preds = self(X_tensor)

            
            # izpiši pomembnosti značilk

            # Find first linear layer and get its weights
            first_linear = None
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    first_linear = layer
                    break

            if first_linear is None:
                print("No linear layer found in model.")
                return preds

            weights = first_linear.weight.data.cpu().numpy()

            if weights.ndim > 1:
                weights = weights.mean(axis=0)


            features = ['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'is_weekend', 'is_holiday', 'is_school_holiday']
            sorted_weights = sorted(zip(weights, features), key=lambda x: abs(x[0]), reverse=True)
        
            print("Feature importances:")
            idx = 0
            for weight, name in sorted_weights:
                print(f"{name:9s}: {weight:7.4f}")
                idx += 1
                if idx > 10:
                    break
            
            
        
        return preds
     
    def get_scaler(self):
        return self.scaler



#def predict_and_fill_test_gaps(test_path, model, output_path="bicikelj_predictions.csv"):
def predict_and_fill_test_gaps(df, model, output_path="bicikelj_predictions.csv"):
    print("****************\nLoading and preparing test data.")

    # Get the scaler from the model
    scaler = model.get_scaler()

    # Load test data manually
    #df = pd.read_csv(test_path)

    # Add base time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    df = add_holiday_features(df)  # adds is_holiday and is_school_holiday

    # One-hot encode hour and month
    hour_ohe = pd.get_dummies(df['hour'], prefix='hour')
    month_ohe = pd.get_dummies(df['month'], prefix='month')
    df = pd.concat([df.drop(columns=['hour', 'month']), hour_ohe, month_ohe], axis=1)

    # Ensure all one-hot columns are present
    expected_hours = [f'hour_{i}' for i in range(24)]
    expected_months = [f'month_{i}' for i in range(1, 13)]
    for col in expected_hours + expected_months:
        if col not in df.columns:
            df[col] = 0

    # Identify station columns
    base_attributes = expected_hours + expected_months + ['is_weekend', 'is_holiday', 'is_school_holiday']
    station_columns = sorted(df.columns.difference(['timestamp'] + base_attributes))

    # Add shifted (previous hour) station values
    # Create shifted columns all at once using a list of DataFrames
    prev_dfs = []
    for h in [1, 2]:
        shifted = df[station_columns].shift(h).add_prefix(f'prev_{h}h_')
        prev_dfs.append(shifted)

    # Combine original DataFrame with shifted features
    df = pd.concat([df] + prev_dfs, axis=1)

    # Drop invalid rows (first row or rows where gap broke continuity)
    df['delta'] = df['timestamp'].diff().dt.total_seconds().div(3600)
    for h in [1, 2]:
        df.loc[df['delta'] != 1.0, [f'prev_{h}h_{s}' for s in station_columns]] = np.nan
    df.drop(columns='delta', inplace=True)

    # Drop rows with missing history
    history_features = [f'prev_{h}h_{s}' for h in [1, 2] for s in station_columns]
    df = df.dropna(subset=history_features)

    # Attributes used for prediction
    #attributes = expected_hours + expected_months + ['is_weekend', 'is_holiday', 'is_school_holiday'] + [f'prev_{s}' for s in station_columns]
    # Rebuild final attribute list
    full_attributes = base_attributes + history_features

    # Fill missing columns in case model was trained on more
    model_input_dim = scaler.mean_.shape[0]
    for col in full_attributes:
        if col not in df.columns:
            df[col] = 0.0

    # Just to be safe, sort attributes in same order
    full_attributes = sorted(full_attributes)


    df_filled = df.copy()

    for idx in range(len(df)):
        if df.iloc[idx][station_columns].isna().all():
            X_row = df.iloc[[idx]][full_attributes].values.astype(np.float32)
            X_scaled = scaler.transform(X_row)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            with torch.no_grad():
                y_pred = model(X_tensor).squeeze(0).numpy()

            df_filled.iloc[idx, df_filled.columns.get_indexer(station_columns)] = y_pred

    # Output only filled predictions
    df_output = df_filled[df[station_columns].isna().all(axis=1)][['timestamp'] + list(station_columns)]
    #df_output['timestamp'] = original_timestamps

    #df_output.to_csv(output_path, index=False)
    #print(f"Saved predictions to {output_path}")

    return df_output






if __name__ == "__main__":

    # prepare data for training
    data = PrepareData("../podatki/bicikelj_train.csv", train_size=TRAIN_SIZE, predict_size=PREDICT_SIZE, batch_size=BATCH_SIZE)

    # get training set
    # extract training data
    X_full, y_full = data.get_train_loader()  # returns tensors

    """
    # convert to numpy for KFold
    X_np = X_full.numpy()
    y_np = y_full.numpy()

    # set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        print(f"\n--- Fold {fold+1} ---")
        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]

        model = Model_Neural_Network()
        model.fit(X_train, y_train)

        y_pred = model.predict(torch.tensor(X_val, dtype=torch.float32))
        loss_fn = nn.MSELoss()
        val_loss = loss_fn(y_pred, torch.tensor(y_val, dtype=torch.float32)).item()
        print(f"Fold {fold+1} MSE: {val_loss:.4f}")
        mse_scores.append(val_loss)

    # summary
    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print(f"\nAverage Validation MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    """
    # FINAL MODEL on all data
    final_model = Model_Neural_Network()
    final_model.fit(X_full, y_full)

    # Evaluate on test split
    X_test, y_test = data.get_test_data()
    y_pred = final_model.predict(X_test)
    final_mse = nn.MSELoss()(y_pred, y_test).item()
    print(f"\nFinal MSE on 20% holdout: {final_mse:.4f}")



    # EVALUATE ON ACTUAL TEST DATA
    #predict_and_fill_test_gaps("../podatki/bicikelj_test.csv", final_model)

    test_data = pd.read_csv("../podatki/bicikelj_test.csv")
    original_timestamps = test_data['timestamp'].copy()
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], utc=True)
    test_data = test_data.sort_values('timestamp').reset_index(drop=True)

    predictions = []
    for step in range(4):
        predicted_row = predict_and_fill_test_gaps(test_data, final_model)

        # insert the predicted row to test_data and ensure, theres one less nan row and still sorted
        
        # add prediction to all predictions. If it's step 0, then keep the header, otherwise not
        predictions.append(predicted_row)
        print(f"predicted row length: {len(predicted_row)}, predictions len: {len(predictions)}")

        # update the test_data and sort it again:
        #test_data = pd.concat([test_data, predicted_row], ignore_index=True)
        #test_data = test_data.sort_values('timestamp').reset_index(drop=True)
        # Match by timestamp and update station values
        for i in range(len(predicted_row)):
            pred_time = predicted_row.iloc[i]['timestamp']
            station_columns = predicted_row.columns.difference(['timestamp'])

            matching_idx = test_data[test_data['timestamp'] == pred_time].index
            if not matching_idx.empty:
                test_data.loc[matching_idx[0], station_columns] = predicted_row.iloc[i][station_columns].values
            else:
                print(f"⚠️ Warning: Could not find matching timestamp {pred_time} in test_data.")

        print("\n--- Final filled rows 49 to 52 ---")
        print(test_data.loc[100:103])

    # Combine all predicted rows into one DataFrame
    final_predictions = pd.concat(predictions, ignore_index=True)

    # sort them:
    final_predictions = final_predictions.sort_values('timestamp').reset_index(drop=True)
    #final_predictions['timestamp'] = original_timestamps
    final_predictions['timestamp'] = final_predictions['timestamp'].dt.strftime('%Y-%m-%dT%H:%MZ')

    final_predictions.to_csv("bicikelj_predictions.csv", index=False)
    print("All 4 predictions completed and saved.")
