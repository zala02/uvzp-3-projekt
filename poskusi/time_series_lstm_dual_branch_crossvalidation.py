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

# additional information about holidays
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from podatki.holidays import add_holiday_features


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slower, but deterministic

set_seed(42)



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


class PrepareData():

    def __init__(self, attributes, time_series, train_size=48, predict_size=4, gap_size=26):

        self.attributes = attributes
        self.time_series = time_series

        self.train_size = train_size
        self.predict_size = predict_size
        self.gap_size = gap_size


        # train test
        self.X_train_timeseries = None
        self.X_train_attributes = None
        self.X_val_timeseries = None
        self.X_val_attributes = None
        self.y_train = None
        self.y_val = None

        # standard scaler
        self.scaler = None
        self.scaler_mean = None 
        self.scaler_std = None 

        self.X_train_standardize = None
        self.X_val_standardize = None
        self.y_train_standardize = None
        self.y_val_standardize = None

        # tensors
        self.X_train_timeseries_tensor = None
        self.X_val_timeseries_tensor = None
        self.X_train_attributes_tensor = None
        self.X_val_attributes_tensor = None
        self.y_train_tensor = None
        self.y_val_tensor = None


    def adjust_attributes(self):
        """keep only rows in attributes whose timestamp appears in time_series"""
        
        # Keep only rows with matching timestamps
        matched_attributes = self.attributes[
            self.attributes["timestamp"].isin(self.time_series["timestamp"])
        ].reset_index(drop=True)

        self.attributes = matched_attributes

    def create_train_test(self, y):
        """create train and val sets"""

        #print("***\nCreating train and test sets started")

        # the percentage division needs to be exactly at 48k and 4k 
        training_ratio = 0.80
        split_train_samples = int(len(self.time_series) * training_ratio)
        split_predict_samples = int(len(y) * training_ratio)

        # separate by index
        X_train_timeseries = self.time_series.iloc[:split_train_samples].copy()
        X_train_attributes = self.attributes.iloc[:split_train_samples].copy()

        X_test_timeseries = self.time_series.iloc[split_train_samples:].copy()
        X_test_attributes = self.attributes.iloc[split_train_samples:].copy()

        y_train = y.iloc[:split_predict_samples].copy()
        y_test = y.iloc[split_predict_samples:].copy()
        #print(f"X_train_timeseries shape: {X_train_timeseries.shape}, X_test_timeseries shape: {X_test_timeseries.shape}")
        #print(f"X_train_attributes shape: {X_train_attributes.shape}, X_test_attributes shape: {X_test_attributes.shape}")
        #print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        #print(f"X: {len(X_train)} + {len(X_test)} = {len(X_train) + len(X_test)}, must be equal to {len(self.X)}")
        #print(f"y: {len(y_train)} + {len(y_test)} = {len(y_train) + len(y_test)}, must be equal to {len(self.y)}")

        #print(f"X train TIME SERIES\n{X_train_timeseries}")
        #print(f"y train TIME SERIES\n{y_train}")

        #print(f"X train ATTRIBUTES\n{X_train_attributes}")
        #print(f"y train ATTRIBUTES\n{y_train}")

        #print(f"X test TIME SERIES\n{X_test_timeseries}")
        #print(f"y test\n{y_test}")

        #print(f"X test ATTRIBUTES\n{X_test_attributes}")
        #print(f"y test\n{y_test}")

        # and finally, store!
        self.X_train_timeseries = X_train_timeseries
        self.X_train_attributes = X_train_attributes
        self.X_val_timeseries = X_test_timeseries
        self.X_val_attributes = X_test_attributes
        self.y_train = y_train
        self.y_val = y_test

        #print("Creating train and test sets finished")

    def standardize_data(self, station_name):
        """panda to numpy + standardize"""

        #print("***\nData standardization started")

        # only need to standardize time series
        self.X_train_standardize = self.X_train_timeseries[[station_name]].values
        self.X_val_standardize = self.X_val_timeseries[[station_name]].values

        # standardize
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train_standardize)
        self.scaler_mean = self.scaler.mean_[0] 
        self.scaler_std = self.scaler.scale_[0] 

        self.X_train_standardize = self.scaler.transform(self.X_train_standardize)
        self.X_val_standardize = self.scaler.transform(self.X_val_standardize)

        # only store y
        self.y_train_standardize = self.y_train[station_name].values
        self.y_val_standardize = self.y_val[station_name].values

        """
        print(self.X_train_standardize)
        print(self.X_train_standardize.shape)

        print(self.X_val_standardize)
        print(self.X_val_standardize.shape)

        print(self.y_train_standardize)
        print(self.y_train_standardize.shape)

        print(self.y_val_standardize)
        print(self.y_val_standardize.shape)
        """
        #print("Data standardization finished")


    def numpy_to_tensor(self):
        """return pytorch tensors"""

        #print("***\nTransformation from numpy to pytorch tensor started")


        # === Time Series Tensors ===
        num_train_samples = len(self.X_train_standardize) // self.train_size
        num_val_samples = len(self.X_val_standardize) // self.train_size

        X_train_timeseries_tensor = torch.tensor(
            self.X_train_standardize[:num_train_samples * self.train_size].reshape(num_train_samples, self.train_size, 1),
            dtype=torch.float32
        )

        X_val_timeseries_tensor = torch.tensor(
            self.X_val_standardize[:num_val_samples * self.train_size].reshape(num_val_samples, self.train_size, 1),
            dtype=torch.float32
        )

        # === Attribute Tensors ===
        # Drop timestamp and convert to numpy
        train_attrs = self.X_train_attributes.drop(columns=["timestamp"]).values
        val_attrs = self.X_val_attributes.drop(columns=["timestamp"]).values

        X_train_attributes_tensor = torch.tensor(
            train_attrs[:num_train_samples * self.train_size].reshape(num_train_samples, self.train_size, -1),
            dtype=torch.float32
        )

        X_val_attributes_tensor = torch.tensor(
            val_attrs[:num_val_samples * self.train_size].reshape(num_val_samples, self.train_size, -1),
            dtype=torch.float32
        )

        # === Target Tensors ===
        y_train_tensor = torch.tensor(
            self.y_train_standardize[:num_train_samples * self.predict_size].reshape(num_train_samples, self.predict_size),
            dtype=torch.float32
        )

        y_val_tensor = torch.tensor(
            self.y_val_standardize[:num_val_samples * self.predict_size].reshape(num_val_samples, self.predict_size),
            dtype=torch.float32
        )

        self.X_train_timeseries_tensor = X_train_timeseries_tensor
        self.X_val_timeseries_tensor = X_val_timeseries_tensor
        self.X_train_attributes_tensor = X_train_attributes_tensor
        self.X_val_attributes_tensor = X_val_attributes_tensor
        self.y_train_tensor = y_train_tensor
        self.y_val_tensor = y_val_tensor


        return (
            X_train_timeseries_tensor,
            X_val_timeseries_tensor,
            X_train_attributes_tensor,
            X_val_attributes_tensor,
            y_train_tensor,
            y_val_tensor
        )

        #print("Transformation from numpy to pytorch tensor finished")

    def get_station_names(self):
        """return station names from data"""
        #station_columns = list(self.data.columns.difference(["timestamp"]))
        #return station_columns
        return [col for col in self.time_series.columns if col != "timestamp"]

    def get_standardization_info(self):
        """return mean and std from scaler"""
        return self.scaler_mean, self.scaler_std

    def get_scaler(self):
        return self.scaler


class Model_Dual_Branch(nn.Module):

    def __init__(self, scaler_mean, scaler_std, attr_input_size, lr=0.01, epochs=150, weight_dec=0.01, lambda_reg=0.0):
        super().__init__()

        # (batch_size, sequence_length, input_size)
        # sequence_length: How many time steps you're giving the LSTM (e.g. 48 hours).
        # input_size: How many features you have per time step.
        # batch_size: How many samples you’re training in parallel.

        # LSTM for time series
        self.input_size=1
        self.hidden_size=128
        self.num_layers=2
        self.output_size=4

        self.LSTM_model = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
            dropout=0.2
        )

        self.attr_input_size = attr_input_size * 48

        # NN for attributes
        self.output_size_nn = 64
        self.NN_model = nn.Sequential(
            nn.Linear(self.attr_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, self.output_size_nn),
            nn.BatchNorm1d(self.output_size_nn),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.final_model = nn.Sequential(
            nn.Linear(self.hidden_size + self.output_size_nn, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, self.output_size),
        )

        # store other variables
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.lr = lr
        self.wd = weight_dec
        self.epochs = epochs


    def forward(self, x_timeseries, x_attributes):
        _, (h_n, _) = self.LSTM_model(x_timeseries)  # h_n: (num_layers, batch, hidden)
        time_out = h_n[-1]  # (batch, hidden_size)

        # Flatten attribute input
        attr_flat = x_attributes.reshape(x_attributes.size(0), -1)  # (batch, 48 * attr_input_size)
        attr_out = self.NN_model(attr_flat)          # (batch, 64)

        # Concatenate both branches
        combined = torch.cat((time_out, attr_out), dim=1)  # (batch, hidden + 64)
        out = self.final_model(combined)  # (batch, output_size=4)
        return out

    
    def fit(self, X_timeseries_tensor, X_attributes_tensor, y_tensor, batch_size=16, patience=20):
        print("************\nTRAINING")

        # batch learning
        dataset = TensorDataset(X_timeseries_tensor, X_attributes_tensor, y_tensor)
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

            for batch_X_time, batch_X_attr, batch_y in dataloader:
                
                pred = self.forward(batch_X_time, batch_X_attr)
                loss = loss_fn(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            if epoch % 30 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f} - Total Loss: {total_loss:.4f}")
                #print(f"Epoch {epoch} - Loss: {loss.item():.4f}")


            # Early stopping check
            if avg_loss < best_loss - 1e-4: 
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    #print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        

    def predict(self, X_timeseries_tensor, X_attributes_tensor):
 
        self.eval()
        with torch.no_grad():
            
            preds_standard = self.forward(X_timeseries_tensor, X_attributes_tensor)
            #preds = preds_standard * self.scaler_std + self.scaler_mean

            preds = np.clip(preds_standard, a_min=0, a_max=None)

            return preds

def cross_validate_time_series(timeseries_class, attributes_class, preparedata_class, batch_size, lr, weight_dec, epoch, patience, lambda_reg, k=5):

    attributes = preparedata_class.attributes
    X, y = timeseries_class.X, timeseries_class.y
    attributes_list = attributes_class.attributes

    fold_size_X = 2544
    fold_size_y = 212

    mae_scores = []

    for fold in range(k):
        fold_start_X = fold * fold_size_X
        fold_start_y = fold * fold_size_y

        # separate by index

        X_train_timeseries = X.iloc[fold_start_X : fold_start_X + 2016].copy()
        X_train_attributes = attributes.iloc[fold_start_X : fold_start_X + 2016].copy()
        #print(X_train_timeseries)
        #print(X_train_timeseries.shape)
        #print(X_train_attributes)
        #print(X_train_attributes.shape)

        X_val_timeseries = X.iloc[fold_start_X + 2016 : fold_start_X + fold_size_X].copy()
        X_val_attributes = attributes.iloc[fold_start_X + 2016 : fold_start_X + fold_size_X].copy()
        #print(X_val_timeseries)
        #print(X_val_timeseries.shape)
        #print(X_val_attributes)
        #print(X_val_attributes.shape)

        y_train = y.iloc[fold_start_y : fold_start_y + 168].copy()
        y_val = y.iloc[fold_start_y + 168 : fold_start_y + fold_size_y].copy()
        #print(y_train)
        #print(y_train.shape)
        #print(y_val)
        #print(y_val.shape)

        """
        X_train_timeseries.to_csv(f"../debugiranje/fold{fold}_X_train_timeseries.csv", index=False)
        X_train_attributes.to_csv(f"../debugiranje/fold{fold}_X_train_attributes.csv", index=False)
        X_val_timeseries.to_csv(f"../debugiranje/fold{fold}_X_val_timeseries.csv", index=False)
        X_val_attributes.to_csv(f"../debugiranje/fold{fold}_X_val_attributes.csv", index=False)
        y_train.to_csv(f"../debugiranje/fold{fold}_y_train.csv", index=False)
        y_val.to_csv(f"../debugiranje/fold{fold}_y_val.csv", index=False)
        """

        preparedata_class.X_train_timeseries = X_train_timeseries
        preparedata_class.X_train_attributes = X_train_attributes
        preparedata_class.X_val_timeseries = X_val_timeseries
        preparedata_class.X_val_attributes = X_val_attributes
        preparedata_class.y_train = y_train
        preparedata_class.y_val = y_val

        # we create NN for each station 
        all_preds = []
        all_targets = []

        # we're gonna test on only three stations, the one in the middle (when scaled):
        #stations = ["GRUDNOVO NABREŽJE-KARLOVŠKA C.", "POVŠETOVA - KAJUHOVA", "HOFER-KAJUHOVA"]
        stations = ["CITYPARK", "POLJANSKA-POTOČNIKOVA", "SITULA"]
        #stations = ["CANKARJEVA UL.-NAMA"]

        for station in stations:
            #print(f"\n*********predicting for station {station}*********")

            # standardize data
            preparedata_class.standardize_data(station)
            scaler_mean, scaler_std = preparedata_class.get_standardization_info()

            # create torches
            X_train_timeseries_tensor, X_val_timeseries_tensor, X_train_attributes_tensor, X_val_attributes_tensor, y_train_tensor, y_val_tensor = preparedata_class.numpy_to_tensor()
            #print(X_train_timeseries_tensor.shape)
            #print(X_val_timeseries_tensor.shape)
            #print(X_train_attributes_tensor.shape)
            #print(X_val_attributes_tensor.shape)
            #print(y_train_tensor.shape)
            #print(y_val_tensor.shape)

            # train the model
            model = Model_Dual_Branch(scaler_mean=scaler_mean, scaler_std=scaler_std, attr_input_size=(len(attributes_list)), lr=lr, epochs=epoch, weight_dec=weight_dec, lambda_reg=lambda_reg)
            model.fit(X_train_timeseries_tensor, X_train_attributes_tensor, y_train_tensor, batch_size=batch_size, patience=patience)

            # test it
            y_pred = model.predict(X_val_timeseries_tensor, X_val_attributes_tensor)
            #y_pred = np.clip(y_pred, a_min=0, a_max=None)

            mae = nn.MSELoss()(y_pred, y_val_tensor).item()
            print(f"MAE on local test data: {mae:.4f}")

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
    timeseries_class.load_and_process()
    X_pandas = timeseries_class.X

    # prepare attributes 
    attributes_class = PrepareAttributres(train_path)
    attributes_class.load_and_process()
    attributes_pandas = attributes_class.processed_data

    preparedata_class = PrepareData(attributes_pandas, X_pandas)
    preparedata_class.adjust_attributes()


    # hyperparams
    EPOCHS = [300]
    BATCH_SIZES = [8, 16, 32]
    LEARNING_RATES = [0.001]
    WEIGHT_DECAYS = [0.01, 0.005, 0.001]
    LAMBDA_REGRESSIONS = [0.0]
    PATIENCES = [10, 20, 30]

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

                            if lr == 0.001 and epoch == 50:
                                continue

                            mae = cross_validate_time_series(
                                timeseries_class=timeseries_class,
                                attributes_class=attributes_class,
                                preparedata_class=preparedata_class,
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




    """
    # create a class for time data manipulation
    train_path = "../podatki/bicikelj_train.csv"

    # prepare time series data
    timeseries_class = PrepareTimeSeries(train_path)
    timeseries_class.load_and_process()
    X_pandas = timeseries_class.X
    y_pandas = timeseries_class.y

    # prepare attribute data
    attributes_class = PrepareAttributres(train_path)
    attributes_class.load_and_process()
    attributes_pandas = attributes_class.processed_data
    attributes_list = attributes_class.attributes

    # prepare whole data
    preparedata_class = PrepareData(attributes_pandas, X_pandas)
    preparedata_class.adjust_attributes()
    preparedata_class.create_train_test(y_pandas)

    stations = preparedata_class.get_station_names()
    mae_results = {}
    all_preds = []
    all_targets = []
    predictions = {}

    for idx, station in enumerate(stations):
        print(f"\n*********predicting for station number {idx+1}: {station}*********")

        # standardize data
        preparedata_class.standardize_data(station)
        scaler_mean, scaler_std = preparedata_class.get_standardization_info()

        # create torches
        X_train_timeseries_tensor, X_val_timeseries_tensor, X_train_attributes_tensor, X_val_attributes_tensor, y_train_tensor, y_val_tensor = preparedata_class.numpy_to_tensor()

        # train the model
        model = Model_Dual_Branch(scaler_mean=scaler_mean, scaler_std=scaler_std, attr_input_size=(len(attributes_list)))
        model.fit(X_train_timeseries_tensor, X_train_attributes_tensor, y_train_tensor)

        # test it
        y_pred = model.predict(X_val_timeseries_tensor, X_val_attributes_tensor)

        mae = nn.MSELoss()(y_pred, y_val_tensor).item()
        print(f"MAE on local test data: {mae:.4f}")

        all_preds.append(y_pred)
        all_targets.append(y_val_tensor)

        
        # actual test data
        test_path = "../podatki/bicikelj_test.csv"
        scaler = timeseries_class.get_scaler()
        final_preds = final_prediction(test_path, station, model, scaler)
        predictions[station] = final_preds.numpy().flatten()  
        

        if idx > 1:
            break

    # final evaluation
    all_preds_tensor = cat(all_preds, dim=0)
    all_targets_tensor = cat(all_targets, dim=0)

    # global MAE
    global_mae = nn.MSELoss()(all_preds_tensor, all_targets_tensor).item()
    print(f"\nFinal MAE: {global_mae:.4f}")

    """

    """
    # putting together predictions
    final_predictions = pd.DataFrame(predictions)

    # Add timestamps
    test_data = pd.read_csv(test_path)
    timestamps = []
    for idx in range(0, len(test_data), 52):
        y_slice = test_data.iloc[(idx + 48):(idx + 48 + 4)]
        timestamps.extend(y_slice["timestamp"].values)

    final_predictions.insert(0, "timestamp", timestamps)

    # Save
    final_predictions.to_csv("bicikelj_predictions_time_series_lstm.csv", index=False)
    print("Final submission saved.")
    """
