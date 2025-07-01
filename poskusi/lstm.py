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

        # add time based attributes (hour and day of week)
        #data['hour'] = data['timestamp'].dt.hour
        #data['dayofweek'] = data['timestamp'].dt.dayofweek

        # Add holiday features
        #data = add_holiday_features(data)

        # one-hot encode hours
        #hour_ohe = pd.get_dummies(data['hour'], prefix='hour')
        #data = pd.concat([data.drop(columns=['hour']), hour_ohe], axis=1)


        # add weekend to data
        #data['is_weekend'] = data['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)



        #self.attributes = list(hour_ohe.columns) + ['is_weekend', 'is_holiday', 'is_school_holiday']
        #self.station_columns = data.columns.difference(['timestamp'] + self.attributes)
        self.station_columns = data.columns.difference(['timestamp'])

        #print(f"Identified {len(self.station_columns)} station columns:")
        #print(self.station_columns)


        # drop Nan values (although theres none in train i think) -> no need for that iguess
        #data = data.dropna(subset=self.station_columns)


        print(f"Loading and processing data finished. For now we have following attributes: {self.attributes}")

        return data

    def _create_train_test(self):
        """Divide data to <train_size> and <predict_size>, drop <gap_size>."""

        
        print("***\nCreating train and test sets started")

        X_list, y_list, ts_list = [], [], []

        total_window = self.train_size + self.predict_size + self.gap_size
        max_index = len(self.data) - total_window

        # we skip <gap_size> values after every <train_size> + <predict_size>
        # for example if we have 48 hours for history, then 4 hours of predicting, then we skip next 10 hours
        step_size = self.train_size + self.predict_size + self.gap_size
        for start_idx in range(0, max_index, step_size):
            hist = self.data.iloc[start_idx : start_idx + self.train_size]
            future = self.data.iloc[start_idx + self.train_size : start_idx + self.train_size + self.predict_size]

            # ensure both input and target windows are complete
            if hist[self.station_columns].isna().values.any() or future[self.station_columns].isna().values.any():
                continue

            # in X we put all the attributes
            #X_sample = hist[self.attributes].values.astype(np.float32)
            X_sample = hist[self.station_columns].values.astype(np.float32)
            
            # in Y we put the output
            #cols_to_drop = ['timestamp'] + self.attributes
            #y_sample = future.drop(columns=cols_to_drop).values.astype(np.float32)
            y_sample = future[self.station_columns].values.astype(np.float32)  # shape: [4, num_stations]

            X_list.append(X_sample)
            y_list.append(y_sample)
            ts_list.append(future['timestamp'].values)

        X = np.array(X_list)  # shape: (num_samples, 48, num_stations)
        y = np.array(y_list)  # shape: (num_samples, 4, num_stations)
        timestamps = np.array(ts_list)  # shape: (num_samples, 4)
        #print(f"X shape: {X.shape}, y shape: {y.shape}, timestamps shape: {timestamps.shape}")

        #print(f"\nFinal X shape: {X.shape}, should be (samples, {self.train_size}, {len(self.attributes)})")
        #print(f"Final y shape: {y.shape}, should be (samples, {self.predict_size}, {len(self.station_columns)})")

        # Split into train and validation sets

        #X_train, X_val, y_train, y_val, timestamps_train, timestamps_val = train_test_split(
        #    X, y, timestamps, test_size=0.2, random_state=42
        #)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        timestamps_train, timestamps_val = timestamps[:split_idx], timestamps[split_idx:]

        
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
        
       
        print("Creating train and test sets finished")

        return X_train, X_val, y_train, y_val, timestamps_train, timestamps_val


    def get_train_loader(self):
        """Creates and returns a PyTorch data object for training."""

        # ce zelimo podatke v paketih, poklicemo to funkcijo
        train_dataset = ToTensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size) # shuffle=True
        return train_loader


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

    def __init__(self, input_len=48, input_dim=84, output_len=4, num_stations=84, lr=0.001, epochs=2000):
        super().__init__()

        self.input_size = input_len * input_dim
        self.output_size = output_len * num_stations
        self.output_len = output_len
        self.num_stations = num_stations

        _hidden_size = 64
        _num_layers = 2
        _dropout = 0.2

        self.model = nn.LSTM(
            input_size=input_dim, 
            hidden_size=_hidden_size,
            num_layers=_num_layers, 
            batch_first=True,
            dropout=_dropout
        )

        self.fc = nn.Linear(_hidden_size, output_len * num_stations)


        self.flatten = nn.Flatten()

        self.lr = lr 
        self.lambda_reg = 0.0
        self.epochs = epochs

        # Add StandardScaler instance
        self.scaler = StandardScaler()  


    def forward(self, x):
        # x shape: [batch_size, 48, input_dim]
        lstm_out, _ = self.model(x)  # lstm_out: [batch_size, 48, hidden_dim]
        last_hidden = lstm_out[:, -1, :]  # Take the last time step

        output = self.fc(last_hidden)  # [batch_size, 4 * 84]
        output = output.view(-1, self.output_len, self.num_stations)  # [batch_size, 4, 84]
        return output

    def soft_threshold(self, param, lmbd):
        with torch.no_grad():
            # postavi na 0 utezi, ki so manjse od lmbd
            # tiste, ko so vecje, pa zmanjsa za lmbd
            # param.copy_() je funkcija, ki lahko neposredno popravi utezi
            param.copy_(param.sign() * torch.clamp(param.abs() - lmbd, min=0.0))

    
    def fit(self, train_loader):
        
        # first we need to standardize data
        all_X = []
        for xb, _ in train_loader:
            all_X.append(xb.numpy())
        all_X = np.concatenate(all_X, axis=0)  # shape: (N, 48, 4)
        flat_X = all_X.reshape(-1, all_X.shape[-1])  # shape: (N*48, 4)
        self.scaler.fit(flat_X)
        

        #optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0

            for xb, yb in train_loader:
                
                # standardize data
                b = xb.shape[0]
                reshaped = xb.view(-1, xb.shape[-1]).numpy()  # shape: (b*48, 4)
                normed = self.scaler.transform(reshaped)
                normed_tensor = torch.tensor(normed, dtype=torch.float32).view(b, 48, -1)
                

                pred = self(normed_tensor)
                mse_loss = loss_fn(pred, yb)

                # add L1 regulariazition
                l1_norm = sum(param.abs().sum() for name, param in self.named_parameters() if 'weight' in name)
                loss = mse_loss + self.lambda_reg * l1_norm

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Apply soft thresholding to weights
                # if you do neural networs, youll need to do the for loop
                for name, param in self.named_parameters():
                    if 'weight' in name:
                        self.soft_threshold(param, self.lambda_reg)

                total_loss += loss.item()
            if epoch % 200 == 0:
                print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    def predict(self, X):
 
        self.eval()
        with torch.no_grad():
            
            
            # standardize data
            b = X.shape[0]
            X_flat = X.view(-1, X.shape[-1]).numpy()
            X_normed = self.scaler.transform(X_flat)
            X_tensor = torch.tensor(X_normed, dtype=torch.float32).view(b, 48, -1)
            
            preds = self(X_tensor)

            """
            # izpiši pomembnosti značilk

            # Find first linear layer and get its weights
            first_linear = None
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    first_linear = layer
                    break

            if first_linear is None:
                print("No linear layer found in model.")
                return

            weights = first_linear.weight.data.cpu().numpy()

            if weights.ndim > 1:
                weights = weights.mean(axis=0)


            features = ['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'is_weekend', 'is_holiday', 'is_school_holiday']
            sorted_weights = sorted(zip(weights, features), key=lambda x: abs(x[0]), reverse=True)
            for weight, name in sorted_weights:
                print(f"{name:9s}: {weight:7.4f}")
            """
        
        return preds
     
    def get_scaler(self):
        return self.scaler



def predict_and_fill_test_gaps(test_path, model, output_path="bicikelj_predictions.csv"):
    print("****************\nLoading and preparing test data.")

    # Get the scaler from the model
    scaler = model.get_scaler()

    # Load test data manually
    df = pd.read_csv(test_path)
    original_timestamps = df['timestamp'].copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')

    # Manually apply same preprocessing as in training
    #df = add_holiday_features(df)
    #df['hour'] = df['timestamp'].dt.hour
    #df['dayofweek'] = df['timestamp'].dt.dayofweek
    #df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)

    #hour_ohe = pd.get_dummies(df['hour'], prefix='hour')
    #df = pd.concat([df.drop(columns=['hour']), hour_ohe], axis=1)


    # Define the attributes you used for training
    #attributes = list(hour_ohe.columns) + ['is_weekend', 'is_holiday', 'is_school_holiday']
    #station_columns = df.columns.difference(['timestamp'] + attributes)
    station_columns = df.columns.difference(['timestamp'])


    df_filled = df.copy()

    i = 0
    while i + TRAIN_SIZE + PREDICT_SIZE <= len(df):
        hist = df.iloc[i:i+TRAIN_SIZE]
        future = df.iloc[i+TRAIN_SIZE:i+TRAIN_SIZE+PREDICT_SIZE]

        # Only predict where future block is completely missing
        if future[station_columns].isna().all().all():
            # Standardize input
            X_input = hist[station_columns].values.astype(np.float32)  # shape: (48, 84)
            X_scaled = scaler.transform(X_input)
            X_tensor = torch.tensor(X_scaled).unsqueeze(0)  # shape: (1, 48, 84)

            # Predict
            with torch.no_grad():
                y_pred = model(X_tensor).squeeze(0).numpy()  # shape: (4, num_stations)

            # Fill in predictions
            for j in range(PREDICT_SIZE):
                df_filled.loc[df_filled.index[i + TRAIN_SIZE + j], station_columns] = y_pred[j]

            i += TRAIN_SIZE + PREDICT_SIZE
        else:
            i += 1

    # Restore original timestamps and drop extra columns
    #df_filled = df_filled.drop(columns=attributes)
    df_filled['timestamp'] = original_timestamps

    predicted_rows = df[station_columns].isna().all(axis=1)
    df_output = df_filled.loc[predicted_rows, ['timestamp'] + list(station_columns)]

    # Save only predictions to CSV
    df_output.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")



if __name__ == "__main__":

    # prepare data for training
    data = PrepareData("../podatki/bicikelj_train.csv", train_size=TRAIN_SIZE, predict_size=PREDICT_SIZE, batch_size=BATCH_SIZE)

    # get training set
    train_loader = data.get_train_loader()

    # create a model and send data to training
    model = Model_Neural_Network()
    model.fit(train_loader)

    # evaluate model 
    X_test, y_test = data.get_test_data()
    y_pred = model.predict(X_test)

    # print loss
    loss_fn = nn.MSELoss()
    val_loss = loss_fn(y_pred, y_test).item()
    print(f"Validation MSE: {val_loss:.4f}")




    # EVALUATE ON ACTUAL TEST DATA
    predict_and_fill_test_gaps("../podatki/bicikelj_test.csv", model)
