# basic libraries
import pandas as pd
import numpy as np
import random

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
from torch import cat


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slower, but deterministic

set_seed(42)


class PrepareTimeSeries:

    #def __init__(self, filename, train_size=48, predict_size=4, gap_size=62):
    def __init__(self, filename, train_size=48, predict_size=4, gap_size=26):

        #print("***\nInitialization of PrepareData started")

        # store the variables
        self.filename = filename
        self.train_size = train_size
        self.predict_size = predict_size
        self.gap_size = gap_size
        
        # variables created in this class
        self.data = None
        self.X = None 
        self.y = None 

        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        # standard scaler
        self.scaler = StandardScaler()
        self.scaler_mean = None 
        self.scaler_std = None 

        self.X_train_standardize = None
        self.X_val_standardize = None
        self.y_train_standardize = None
        self.y_val_standardize = None

        # tensors
        self.X_train_tensor = None
        self.X_val_tensor = None
        self.y_train_tensor = None
        self.y_val_tensor = None

        #print("Successfully initialized class PrepareData")

    def load_and_process(self):
        """Load data, drop nan values, add atributes. Divide data to <train_size> and <predict_size>, drop <gap_size>."""

        #print("***\nLoading and processing data started")

        # read the data
        data = pd.read_csv(self.filename)

        # drop the nan rows if theres any
        data = data.dropna(axis=0, how='any', ignore_index=True)

        # save data
        self.data = data

        # we need to split data into X series and y series, dropping gap size after them
        X_list = []
        y_list = []

        total_window = self.train_size + self.predict_size + self.gap_size
        max_index = len(self.data) - total_window

        step_size = self.train_size + self.predict_size + self.gap_size - 1
        for idx in range(0, max_index, step_size):
            
            #  0 : 47, skip 62, 113:160, skip 62 etc
            X_slice = self.data.iloc[idx : (idx + self.train_size)]
            # 47 : 51, skip 62, 161:164, skip 62 etc
            y_slice = self.data.iloc[(idx + self.train_size) : (idx + self.train_size + self.predict_size)]

            #print(f"FIRST 48 HOURS:\n{X_slice}")
            #print(f"NEXT 4 HOURS WE'RE PREDICTING:\n{y_slice}")

            X_list.append(X_slice)
            y_list.append(y_slice)

        
        X = pd.concat(X_list, ignore_index=True)
        y = pd.concat(y_list, ignore_index=True)

        #print(f"X shape: {X.shape}, y shape: {y.shape}")
        #print(f"X values\n{X}")
        #print(f"y values\n{y}")

        self.X = X 
        self.y = y

        #print(f"Loading and processing data finished.")

    def create_train_test(self):
        """create train and val sets"""

        #print("***\nCreating train and test sets started")

        # the percentage division needs to be exactly at 48k and 4k 
        training_ratio = 0.80
        split_train_samples = int(len(self.X) * training_ratio)
        split_predict_samples = int(len(self.y) * training_ratio)
        #print(f"all train samples: {len(self.X)}, they split at: {split_train_samples}")
        #print(f"all test samples: {len(self.y)}, they split at: {split_predict_samples}")

        # separate by index
        X_train = self.X.iloc[:split_train_samples].copy()
        X_test = self.X.iloc[split_train_samples:].copy()
        y_train = self.y.iloc[:split_predict_samples].copy()
        y_test = self.y.iloc[split_predict_samples:].copy()
        #print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        #print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        #print(f"X: {len(X_train)} + {len(X_test)} = {len(X_train) + len(X_test)}, must be equal to {len(self.X)}")
        #print(f"y: {len(y_train)} + {len(y_test)} = {len(y_train) + len(y_test)}, must be equal to {len(self.y)}")

        #print(f"X train\n{X_train}")
        #print(f"y train\n{y_train}")
        #print(f"X test\n{X_test}")
        #print(f"y test\n{y_test}")

        # and finally, store!
        self.X_train = X_train
        self.X_val = X_test
        self.y_train = y_train
        self.y_val = y_test

        #print("Creating train and test sets finished")

    def standardize_data(self, station_name):
        """panda to numpy + standardize"""

        #print("***\nData standardization started")

        self.X_train_standardize = self.X_train[station_name].values.reshape(-1, 1)
        self.X_val_standardize = self.X_val[station_name].values.reshape(-1, 1)
        self.y_train_standardize = self.y_train[station_name].values.reshape(-1, 1)
        self.y_val_standardize = self.y_val[station_name].values.reshape(-1, 1)

        self.scaler.fit(self.X_train[station_name].values.reshape(-1, 1))
        self.scaler_mean = self.scaler.mean_[0] 
        self.scaler_std = self.scaler.scale_[0] 

        self.X_train_standardize = self.scaler.transform(self.X_train_standardize).flatten()
        self.X_val_standardize = self.scaler.transform(self.X_val_standardize).flatten()
        #self.y_train_standardize = self.scaler.transform(self.y_train_standardize).flatten()
        #self.y_val_standardize = self.scaler.transform(self.y_val_standardize).flatten()

        #print("Data standardization finished")


    def numpy_to_tensor(self):
        """return pytorch tensors"""

        #print("***\nTransformation from numpy to pytorch tensor started")


        self.X_train_tensor = torch.tensor(self.X_train_standardize, dtype=torch.float32).view(-1, self.train_size)
        self.X_val_tensor = torch.tensor(self.X_val_standardize, dtype=torch.float32).view(-1, self.train_size)
        self.y_train_tensor = torch.tensor(self.y_train_standardize, dtype=torch.float32).view(-1, self.predict_size) # shape (num_samples, 4)
        self.y_val_tensor = torch.tensor(self.y_val_standardize, dtype=torch.float32).view(-1, self.predict_size)

        #print("Transformation from numpy to pytorch tensor finished")

        return self.X_train_tensor, self.X_val_tensor, self.y_train_tensor, self.y_val_tensor

    def get_station_names(self):
        """return station names from data"""
        #station_columns = list(self.data.columns.difference(["timestamp"]))
        #return station_columns
        return [col for col in self.data.columns if col != "timestamp"]

    def get_standardization_info(self):
        """return mean and std from scaler"""
        return self.scaler_mean, self.scaler_std

    def get_scaler(self):
        return self.scaler


class Model_LSTM(nn.Module):

    def __init__(self, scaler_mean, scaler_std, lr=0.001, epochs=300, weight_dec=0.001):
        super().__init__()

        # (batch_size, sequence_length, input_size)
        # sequence_length: How many time steps you're giving the LSTM (e.g. 48 hours).
        # input_size: How many features you have per time step.
        # batch_size: How many samples you’re training in parallel.

        self.input_size=1
        self.hidden_size=128
        self.num_layers=2
        self.output_size=4

        self.model = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # store other variables
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.lr = lr
        self.wd = weight_dec
        self.epochs = epochs


    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.model(x)
        # use the final time step
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        #out = torch.relu(self.fc(out))

        return out

    
    def fit(self, X_tensor, y_tensor, batch_size=64, patience=10):
        #print("************\nTRAINING")

        # batch learning
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 3. Define optimizer and loss
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0

        # 4. Training loop
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0

            for batch_X, batch_y in dataloader:
                
                #print(type(self.model))
                pred = self.forward(batch_X.unsqueeze(-1))
                #print(type(pred))


                loss = loss_fn(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            #if epoch % 20 == 0 or epoch == self.epochs - 1:
            #    print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f} - Total Loss: {total_loss:.4f}")
            #    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

            # Early stopping check
            if avg_loss < best_loss - 1e-4:  # small delta
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    #print(f"Early stopping triggered at epoch {epoch+1}")
                    break


    def predict(self, X_tensor):
 
        self.eval()
        with torch.no_grad():
            
            preds_standard = self(X_tensor.unsqueeze(-1))
            #preds = preds_standard * self.scaler_std + self.scaler_mean

            preds = np.clip(preds_standard, a_min=0, a_max=None)

            return preds


def cross_validate_time_series(timeseries_class, batch_size, lr, weight_dec, epoch, patience, k=5):

    X, y = timeseries_class.X, timeseries_class.y
    fold_size_X = 2544
    fold_size_y = 212

    mae_scores = []

    for fold in range(k):
        fold_start_X = fold * fold_size_X
        fold_start_y = fold * fold_size_y

        # separate by index
        X_train = X.iloc[fold_start_X : fold_start_X + 2016].copy()
        X_val = X.iloc[fold_start_X + 2016 : fold_start_X + fold_size_X].copy()
        y_train = y.iloc[fold_start_y : fold_start_y + 168].copy()
        y_val = y.iloc[fold_start_y + 168 : fold_start_y + fold_size_y].copy()

        timeseries_class.X_train = X_train
        timeseries_class.X_val = X_val
        timeseries_class.y_train = y_train
        timeseries_class.y_val = y_val

        # we create NN for each station 
        all_preds = []
        all_targets = []

        # we're gonna test on only three stations, the one in the middle (when scaled):
        #stations = ["GRUDNOVO NABREŽJE-KARLOVŠKA C.", "POVŠETOVA - KAJUHOVA", "HOFER-KAJUHOVA"]
        stations = ["CITYPARK", "POLJANSKA-POTOČNIKOVA", "SITULA"]
        #stations = ["CANKARJEVA UL.-NAMA"]
        #stations = ["CITYPARK"]

        for station in stations:
            #print(f"\n*********predicting for station {station}*********")

            # standardize data
            timeseries_class.standardize_data(station)

            # create torches
            X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = timeseries_class.numpy_to_tensor()

            # time for NN
            mean, std = timeseries_class.get_standardization_info()

            # train the model
            model = Model_LSTM(scaler_mean=mean, scaler_std=std, lr=lr, epochs=epoch, weight_dec=weight_dec)
            model.fit(X_train_tensor, y_train_tensor, batch_size=batch_size, patience=patience)

            # test it
            y_pred = model.predict(X_val_tensor)

            mae = nn.MSELoss()(y_pred, y_val_tensor).item()
            #print(f"MAE on local test data: {mae:.4f}")

            all_preds.append(y_pred)
            all_targets.append(y_val_tensor)

        # final evaluation
        all_preds_tensor = cat(all_preds, dim=0)
        all_targets_tensor = cat(all_targets, dim=0)

        # global MAE
        global_mae = nn.MSELoss()(all_preds_tensor, all_targets_tensor).item()
        #print(f"\nFor fold {fold} MAE is {global_mae:.4f}")
        mae_scores.append(global_mae)

    return np.mean(mae_scores)

if __name__ == "__main__":



    # create a class for time data manipulation
    train_path = "../podatki/bicikelj_train.csv"
    timeseries_class = PrepareTimeSeries(train_path)

    # prepare data (drop nan, 48h+4h+gap etc)
    timeseries_class.load_and_process()
    # in X we have [8640 rows x 85 columns], and in y [720 rows x 85 columns]

    # create train test sets
    #timeseries_class.create_train_test()



    # hyperparams
    EPOCHS = [300]
    BATCH_SIZES = [8]
    LEARNING_RATES = [0.01, 0.05, 0.001]
    WEIGHT_DECAYS = [0.01, 0.005, 0.001 ]
    PATIENCES = [20]

    model_idx = 1
    best_mae = float("inf")
    best_config = None

    all_results = []  

    for epoch in EPOCHS:
        for batch_size in BATCH_SIZES:
            for lr in LEARNING_RATES:
                for weight_dec in WEIGHT_DECAYS:
                    for patience in PATIENCES:

                        mae = cross_validate_time_series(
                            timeseries_class=timeseries_class,
                            batch_size=batch_size,
                            lr=lr,
                            weight_dec=weight_dec,
                            epoch=epoch,
                            patience=patience
                        )

                        result = {
                            "model": model_idx,
                            "batch_size": batch_size,
                            "lr": lr,
                            "weight_decay": weight_dec,
                            "epochs": epoch,
                            "patience": patience,
                            "mae": mae
                        }
                        all_results.append(result)

                        print(f"\n****** MODEL {model_idx} ******")
                        print(f"batch={batch_size}, lr={lr}, wd={weight_dec}, epochs={epoch}, patience={patience}")
                        print(f"-> MAE: {mae:.4f}")

                        if mae < best_mae:
                            best_mae = mae
                            best_config = (batch_size, lr, weight_dec, epoch, patience)

                        model_idx += 1

    print("\nBest configuration:")
    print(f"batch size: {best_config[0]}, lr: {best_config[1]}, wd: {best_config[2]}, epochs: {best_config[3]}, patience: {best_config[4]}")
    print(f"Best MSE: {best_mae:.4f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df_sorted = results_df.sort_values(by="mae").reset_index(drop=True)
    results_df_sorted.to_csv("cross_validation_results_lstm.csv", index=False)


