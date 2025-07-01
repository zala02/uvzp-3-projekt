# basic libraries
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
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

        # Ensure full one-hot coverage
        for i in range(24):
            col = f"hour_{i}"
            if col not in hour_ohe.columns:
                hour_ohe[col] = 0

        for i in range(1, 13):
            col = f"month_{i}"
            if col not in month_ohe.columns:
                month_ohe[col] = 0

        for i in range(7):
            col = f"day_of_week_{i}"
            if col not in day_of_week_ohe.columns:
                day_of_week_ohe[col] = 0

        # Align column order (very important!)
        hour_ohe = hour_ohe[[f"hour_{i}" for i in range(24)]]
        month_ohe = month_ohe[[f"month_{i}" for i in range(1, 13)]]
        day_of_week_ohe = day_of_week_ohe[[f"day_of_week_{i}" for i in range(7)]]

        data = pd.concat([data.drop(columns=['hour', 'month', 'day_of_week']), hour_ohe, month_ohe, day_of_week_ohe], axis=1)

        #print(data)

        attributes = ['is_holiday', 'is_school_holiday'] + list(hour_ohe.columns) + list(month_ohe.columns) + list(day_of_week_ohe)
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

        # X data with attributes added
        self.X_train_w_attributes = None
        self.X_val_w_attributes = None

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

        #print(f"X train\n{self.X_train[station_name]}")
        #print(f"y train\n{self.X_val[station_name]}")
        #print(f"X test\n{self.y_train[station_name]}")
        #print(f"y test\n{self.y_val[station_name]}")

        #print(f"X train\n{self.X_train[station_name].values.reshape(-1, 1)}")
        #print(f"y train\n{self.X_val[station_name].values.reshape(-1, 1)}")
        #print(f"X test\n{self.y_train[station_name].values.reshape(-1, 1)}")
        #print(f"y test\n{self.y_val[station_name].values.reshape(-1, 1)}")

        attributes = attributes_class.processed_data  # contains 'timestamp' and all attributes

        def add_attributes(X, station_name):
            # Keep only the relevant station column and timestamp
            X_temp = X[[station_name]].copy()
            X_temp["timestamp"] = X["timestamp"].values

            # Merge on timestamp
            merged = pd.merge(X_temp, attributes, on="timestamp", how="left")

            # Drop timestamp if not needed
            return merged.drop(columns=["timestamp"])

        self.X_train_w_attributes = add_attributes(self.X_train, station_name)
        self.X_val_w_attributes = add_attributes(self.X_val, station_name)

        #print(self.X_train_w_attributes)
        #print(self.X_val_w_attributes)


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
        #print(self.X_train_standardize.shape)
        #print(self.X_val_standardize.shape)

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

    def __init__(self, scaler_mean, scaler_std, attributes_size, lr=0.01, epochs=300, weight_dec=0.01):
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

        #self.fc = nn.Linear(self.hidden_size, self.output_size)
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


    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.model(x)
        # use the final time step
        out = lstm_out[:, -1, :]
        #out = self.norm(out)  
        #out = self.fc(out)
        out = self.head(out)

        return out

    
    def fit(self, X_tensor, y_tensor, batch_size=16, patience=15):
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
            if avg_loss < best_loss - 1e-4:  # small delta
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        

    def predict(self, X_tensor):
 
        self.eval()
        with torch.no_grad():
            
            preds_standard = self(X_tensor)
            #preds = preds_standard * self.scaler_std + self.scaler_mean

            preds = np.clip(preds_standard, a_min=0, a_max=None)

            return preds

def final_prediction(test_path, station, model, scaler, output_path="bicikelj_predictions_lstm.csv"):

    data = pd.read_csv(test_path)

    attributes_class_test = PrepareAttributres(test_path)
    attributes_class_test.load_and_process()
    attributes_data_test = attributes_class_test.processed_data
    attributes_test = attributes_class_test.attributes
    #print(attributes_test)

    # get X and y
    X_list = []

    for idx in range(0, len(data), 52):
        
        #  0 : 47, skip 62, 113:160, skip 62 etc
        X_slice = data.iloc[idx : (idx + 48)]
        # 47 : 51, skip 62, 161:164, skip 62 etc

        X_list.append(X_slice)

    
    X = pd.concat(X_list, ignore_index=True)

    # Merge bike counts with attributes on timestamp
    X_temp = X[[station]].copy()
    X_temp["timestamp"] = X["timestamp"].values
    X_merged = pd.merge(X_temp, attributes_data_test, on="timestamp", how="left")
    X_merged = X_merged.drop(columns=["timestamp"])

    # Standardize only the bike count
    X_bikes = X_merged[[station]].values
    X_bikes_scaled = scaler.transform(X_bikes)

    # Combine with attributes
    X_attrs = X_merged.drop(columns=[station]).values
    X_combined = np.concatenate([X_bikes_scaled, X_attrs], axis=1)

    # Reshape to LSTM input shape
    num_samples = len(X_combined) // 48
    X_tensor = torch.tensor(
        X_combined[:num_samples * 48].reshape(num_samples, 48, -1),
        dtype=torch.float32
    )

    # predict
    y_pred = model.predict(X_tensor)

    return y_pred


if __name__ == "__main__":
    """
    # create a class for time data manipulation
    train_path = "../podatki/bicikelj_train.csv"

    # prepare attributes 
    attributes_class = PrepareAttributres(train_path)
    attributes_class.load_and_process()

    # prepare data (drop nan, 48h+4h+gap etc)
    timeseries_class = PrepareTimeSeries(train_path)
    timeseries_class.load_and_process()
    # in X we have [8640 rows x 85 columns], and in y [720 rows x 85 columns]

    # create train test sets
    timeseries_class.create_train_test()

    # we create NN for each station 
    # or at least ill create a nn for one station then well see
    stations = timeseries_class.get_station_names()
    #print(stations)

    mae_results = {}
    all_preds = []
    all_targets = []
    predictions = {}

    for idx, station in enumerate(stations):
        print(f"\n*********predicting for station number {idx+1}: {station}*********")

        # concat number of bikes with other attributes
        timeseries_class.concat_datas(station, attributes_class)

        # standardize data
        timeseries_class.standardize_data(station)

        # create torches
        X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = timeseries_class.numpy_to_tensor()



        # time for NN
        mean, std = timeseries_class.get_standardization_info()
        print(f"mean: {mean}, std: {std}")

        # train the model
        input_size = X_train_tensor.shape[2]
        #print(f"INPUT SIZE: {input_size}")
        model = Model_LSTM(scaler_mean=mean, scaler_std=std, attributes_size=input_size)
        model.fit(X_train_tensor, y_train_tensor)

        # test it
        y_pred = model.predict(X_val_tensor)

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




    # create a class for time data manipulation
    train_path = "../podatki/bicikelj_train.csv"
    timeseries_class = PrepareTimeSeries(train_path)

    # prepare data (drop nan, 48h+4h+gap etc)
    timeseries_class.load_and_process()
    # in X we have [8640 rows x 85 columns], and in y [720 rows x 85 columns]

    # prepare attributes 
    attributes_class = PrepareAttributres(train_path)
    attributes_class.load_and_process()
    attributes_list = attributes_class.attributes


    X = timeseries_class.X 
    y = timeseries_class.y 

    stations = timeseries_class.get_station_names()
    #print(stations)
    predictions = {}

    for idx, station in enumerate(stations):
        print(f"\n*********predicting for station number {idx+1}: {station}*********")

        # Merge X with attributes on timestamp
        X_temp = X[[station]].copy()
        X_temp["timestamp"] = X["timestamp"].values
        attributes = attributes_class.processed_data
        X_merged = pd.merge(X_temp, attributes, on="timestamp", how="left").drop(columns=["timestamp"])

        # Standardize only bike count
        X_bike_column = X_merged[[station]].values
        scaler = StandardScaler()
        scaler.fit(X_bike_column)
        scaler_mean = scaler.mean_[0] 
        scaler_std = scaler.scale_[0] 
        X_bike_standardized = scaler.transform(X_bike_column)

        # Combine with attributes
        X_attrs = X_merged.drop(columns=[station]).values
        X_standardize = np.concatenate([X_bike_standardized, X_attrs], axis=1)

        # Prepare y
        y_column = y[station].values

        # create torches
        batch_size_X = len(X_standardize) // 48     # 10176 % 48 = 212
        X_seq = X_standardize[:batch_size_X * 48].reshape(batch_size_X, 48, -1)  # (212, 48, 46)
        y_seq = y_column[:batch_size_X * 4].reshape(batch_size_X, 4)  # (212, 4)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        # train the model
        input_size = X_tensor.shape[2]
        model = Model_LSTM(scaler_mean=scaler_mean, scaler_std=scaler_std, attributes_size=input_size)
        model.fit(X_tensor, y_tensor)

        """
        background = X_tensor  # (n, 48, 1)
        #explain_data = X_tensor[0].unsqueeze(-1)  # (50, 48, 1)
        explain_data = X_tensor  # Just one sequence to explain (shape: [1, 48, 1])

        print(f"background shape: {background.shape}")
        print(f"explain_data shape: {explain_data.shape}")

        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explain_data)
        print(f"shap values : {shap_values.shape}")

        sample_idx = 0
        output_hour = 0

        # Extract SHAP values for that sample
        #shap_vals = shap_values[output_hour][:, sample_idx, 0]  # shape: (48,)
        #input_vals = explain_data[sample_idx].squeeze().numpy()  # shape: (48,)
        shap_vals = shap_values[sample_idx, :, :, output_hour]  # shape: (48, 46)
        input_vals = explain_data[0].numpy()     # shape: (48, 46)
        print(f"shap vals: {shap_vals.shape}")
        print(f"input vals: {input_vals.shape}")

        # Flatten for bar plot if you want overall importance per feature
        shap_vals_flat = shap_vals.mean(axis=0)
        input_vals_flat = input_vals.mean(axis=0)

        print(f"shap vals: {shap_vals_flat.shape}")
        print(f"input vals: {input_vals_flat.shape}")

        # Create the Explanation object
        feature_names = ["bike_count"] + attributes_list
        explanation = shap.Explanation(
            values=shap_vals_flat,
            data=input_vals_flat,
            feature_names=feature_names
        )

        # Plot the local bar chart
        plt.title(f"SHAP za postajo {station}")
        shap.plots.bar(explanation, max_display=10)
        """


        # actual test data
        test_path = "../podatki/bicikelj_test.csv"
        final_preds = final_prediction(test_path, station, model, scaler)
        predictions[station] = final_preds.numpy().flatten()  
        #print(final_preds)

        #if idx > 1:
        #    break

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
    final_predictions.to_csv("bicikelj_predictions_time_series_lstm_joined_features.csv", index=False)
    print("Final submission saved.")

