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
from torch import cat

# additional information about holidays
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from podatki.holidays import add_holiday_features

class PrepareAttributres():
    
    def __init__(self, filename):
        
        # store variables
        self.filename = filename

        # other variables
        self.processed_data = None
        self.attributes = []
    
    def load_and_process(self):
        """Prepares attributes for all timestamps, then use whatever u need"""

        print("***\nLoading and processing attributes started")

        data = pd.read_csv(self.filename)
        
        # add holidays
        data = add_holiday_features(data)

        # remove stations:
        data = data[['timestamp', 'is_holiday', 'is_school_holiday']]

        # add one-hot hour, month and day in week
        data['hour'] = data['timestamp'].dt.hour
        data['month'] = data['timestamp'].dt.month
        data['day_of_week'] = data['timestamp'].dt.dayofweek

        hour_ohe = pd.get_dummies(data['hour'], prefix='hour').astype(int)
        month_ohe = pd.get_dummies(data['month'], prefix='month').astype(int)
        day_of_week_ohe = pd.get_dummies(data['day_of_week'], prefix='day_of_week').astype(int)

        data = pd.concat([data.drop(columns=['hour', 'month', 'day_of_week']), hour_ohe, month_ohe, day_of_week_ohe], axis=1)

        #print(data)

        attributes = list(hour_ohe.columns) + list(month_ohe.columns) + list(day_of_week_ohe) + ['is_holiday', 'is_school_holiday']
        #print(f"ADDITIONAL ATTRIBUTES:\n{attributes}")


        # revert time to original + store variables
        data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%dT%H:%MZ')
        self.processed_data = data 
        self.attributes = attributes

        print(f"Loading and processing attributes finished.\n***")


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

    def concat_datas(self, station_name, attributes_class):
        """add attributes from attributes_class to x train and x val for station station_name"""

        attributes = attributes_class.processed_data  # contains 'timestamp' and all attributes

        def add_attributes(X):
            # Keep only the relevant station column and timestamp
            X_temp = X[[station_name]].copy()
            X_temp["timestamp"] = X["timestamp"].values

            # Merge on timestamp
            merged = pd.merge(X_temp, attributes, on="timestamp", how="left")

            # Drop timestamp if not needed
            return merged.drop(columns=["timestamp"])

        self.X_train_w_attributes = add_attributes(self.X_train)
        self.X_val_w_attributes = add_attributes(self.X_val)


    def standardize_data(self, station_name):
        """panda to numpy + standardize"""

        #print("***\nData standardization started")

        # from X with attributes we need to extract station and normalize only that
        X_train_column = self.X_train_w_attributes[[station_name]].values
        X_val_column = self.X_val_w_attributes[[station_name]].values

        # we don't standardize y values
        self.y_train_standardize = self.y_train[station_name].values
        self.y_val_standardize = self.y_val[station_name].values

        # standardize X 
        self.scaler.fit(X_train_column)
        self.scaler_mean = self.scaler.mean_[0] 
        self.scaler_std = self.scaler.scale_[0] 

        X_train_standardized_column = self.scaler.transform(X_train_column)
        X_val_standardized_column = self.scaler.transform(X_val_column)

        # in X we have now only bike count, we need to add attributes again
        X_train_other_attributes = self.X_train_w_attributes.drop(columns=[station_name]).values
        X_val_other_attributes = self.X_val_w_attributes.drop(columns=[station_name]).values
        self.X_train_standardize = np.concatenate([X_train_standardized_column, X_train_other_attributes], axis=1)  # (10176, 46)
        self.X_val_standardize = np.concatenate([X_val_standardized_column, X_val_other_attributes], axis=1)        # (2544, 46)

        #print("Data standardization finished")


    def numpy_to_tensor(self):
        """return pytorch tensors"""

        #print("***\nTransformation from numpy to pytorch tensor started")

        # LSTM mora bit v taki obliki:
        # (batch_size, sequence_length, input_size)
        # batch_size: stevilo zaporedij (torej len(X_train_standardize) // 48)
        # sequence_length: kok je zaporedje dolgo (torej 48 ur)
        # input_size: kok atributov mamo za vsak element (bike counts + atributi = 46)

        batch_size_train = len(self.X_train_standardize) // self.train_size     # 10176 % 48 = 212
        batch_size_val = len(self.X_val_standardize) // self.train_size         # 2544 % 48 = 53

        # reshape inputs: (num_sequences, 48, 46)
        X_train_seq = self.X_train_standardize[:batch_size_train * self.train_size].reshape(batch_size_train, self.train_size, -1)  # (212, 48, 46)
        X_val_seq = self.X_val_standardize[:batch_size_val * self.train_size].reshape(batch_size_val, self.train_size, -1)          # (53, 48, 46)

        # reshape targets: (num_sequences, 4)
        y_train_seq = self.y_train_standardize[:batch_size_train * self.predict_size].reshape(batch_size_train, self.predict_size)  # (212, 4)
        y_val_seq = self.y_val_standardize[:batch_size_val * self.predict_size].reshape(batch_size_val, self.predict_size)          # (53, 4)


        # convert to tensors
        self.X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        self.X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
        self.y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)

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

    def __init__(self, scaler_mean, scaler_std, attributes_size, lr=0.01, epochs=120, weight_dec=0.01, lambda_reg=0.0):
        super().__init__()

        # (batch_size, sequence_length, input_size)
        # sequence_length: How many time steps you're giving the LSTM (e.g. 48 hours).
        # input_size: How many features you have per time step.
        # batch_size: How many samples you’re training in parallel.

        self.input_size=attributes_size 
        self.hidden_size=128
        self.num_layers=2
        self.output_size=4

        self.model = nn.LSTM(
            input_size=self.input_size,     # 46
            hidden_size=self.hidden_size,   # 128
            num_layers=self.num_layers,     # 2
            batch_first=True,
            dropout=0.2
        )

        #self.norm = nn.LayerNorm(self.hidden_size)

        #self.fc = nn.Linear(
        #    self.hidden_size, 
        #    self.output_size
        #)

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            #nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(64, self.output_size)
        )


        # store other variables
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.lr = lr
        self.wd = weight_dec
        self.epochs = epochs
        self.lambda_reg = lambda_reg


    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.model(x)
        # use the final time step
        out = lstm_out[:, -1, :]
        #out = self.fc(out)
        #out = torch.relu(self.fc(out))
        out = self.head(out)

        return out


    def soft_threshold(self, param, lmbd):
        with torch.no_grad():
            # postavi na 0 utezi, ki so manjse od lmbd
            # tiste, ko so vecje, pa zmanjsa za lmbd
            # param.copy_() je funkcija, ki lahko neposredno popravi utezi
            param.copy_(param.sign() * torch.clamp(param.abs() - lmbd, min=0.0))

    
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
                pred = self.forward(batch_X)
                #print(type(pred))


                #loss = loss_fn(pred, batch_y)
                mse_loss = loss_fn(pred, batch_y)

                # add L1 regularization
                l1_norm = sum(param.abs().sum() for name, param in self.named_parameters() if 'weight' in name)
                loss = mse_loss + self.lambda_reg * l1_norm

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
            
            preds_standard = self(X_tensor)
            #preds = preds_standard * self.scaler_std + self.scaler_mean

            preds = np.clip(preds_standard, a_min=0, a_max=None)

            return preds


def cross_validate_time_series(timeseries_class, attributes_class, batch_size, lr, weight_dec, epoch, patience, lambda_reg, k=5):

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

        for station in stations:
            #print(f"\n*********predicting for station {station}*********")

            # concat number of bikes with other attributes
            timeseries_class.concat_datas(station, attributes_class)

            # standardize data
            timeseries_class.standardize_data(station)

            # create torches
            X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = timeseries_class.numpy_to_tensor()

            # time for NN
            mean, std = timeseries_class.get_standardization_info()

            # train the model
            input_size = X_train_tensor.shape[2]
            model = Model_LSTM(scaler_mean=mean, scaler_std=std, attributes_size=input_size, lr=lr, epochs=epoch, weight_dec=weight_dec, lambda_reg=lambda_reg)
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

    # prepare attributes 
    attributes_class = PrepareAttributres(train_path)
    attributes_class.load_and_process()

    # prepare data (drop nan, 48h+4h+gap etc)
    timeseries_class.load_and_process()
    # in X we have [8640 rows x 85 columns], and in y [720 rows x 85 columns]

    # create train test sets
    #timeseries_class.create_train_test()



    # hyperparams
    EPOCHS = [50, 100, 200]
    BATCH_SIZES = [8, 16]
    LEARNING_RATES = [0.01, 0.005]
    WEIGHT_DECAYS = [0.01, 0.005]
    PATIENCES = [300]
    LAMBDA_REGRESSIONS = [0.01, 0.0001, 0.001, 0.00001, 0.0]

    model_idx = 1
    best_mae = float("inf")
    best_config = None

    all_results = []  

    for epoch in EPOCHS:
        for batch_size in BATCH_SIZES:
            for lr in LEARNING_RATES:
                for weight_dec in WEIGHT_DECAYS:
                    for lambda_reg in LAMBDA_REGRESSIONS:
                        for patience in PATIENCES:

                            mae = cross_validate_time_series(
                                timeseries_class=timeseries_class,
                                attributes_class=attributes_class,
                                batch_size=batch_size,
                                lr=lr,
                                weight_dec=weight_dec,
                                epoch=epoch,
                                patience=patience,
                                lambda_reg=lambda_reg
                            )

                            result = {
                                "model": model_idx,
                                "batch_size": batch_size,
                                "lr": lr,
                                "weight_decay": weight_dec,
                                "epochs": epoch,
                                "patience": patience,
                                "lambda_reg": lambda_reg,
                                "mae": mae
                            }
                            all_results.append(result)

                            print(f"\n****** MODEL {model_idx} ******")
                            print(f"batch={batch_size}, lr={lr}, wd={weight_dec}, epochs={epoch}, patience={patience}, lambda reg={lambda_reg}")
                            print(f"-> MAE: {mae:.4f}")

                            if mae < best_mae:
                                best_mae = mae
                                best_config = (batch_size, lr, weight_dec, lambda_reg, epoch, patience)

                            model_idx += 1

    print("\nBest configuration:")
    print(f"batch size: {best_config[0]}, lr: {best_config[1]}, wd: {best_config[2]}, lambda reg={best_config[3]}, epochs: {best_config[4]}, patience: {best_config[5]}")
    print(f"Best MSE: {best_mae:.4f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df_sorted = results_df.sort_values(by="mae").reset_index(drop=True)
    results_df_sorted.to_csv("cross_validation_results_lstm_joined_features.csv", index=False)


