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

# shap
import shap
import matplotlib.pyplot as plt



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

    def __init__(self, scaler_mean, scaler_std, lr=0.01, epochs=100, weight_dec=0.005):
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

    
    def fit(self, X_tensor, y_tensor, batch_size=8, patience=15):
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
            
            preds_standard = self(X_tensor.unsqueeze(-1))
            #preds = preds_standard * self.scaler_std + self.scaler_mean

            preds = np.clip(preds_standard, a_min=0, a_max=None)

            return preds

def final_prediction(test_path, station, model, scaler, output_path="bicikelj_predictions_lstm.csv"):

    data = pd.read_csv(test_path)

    # get X and y
    X_list = []
    y_list = []

    for idx in range(0, len(data), 52):
        
        #  0 : 47, skip 62, 113:160, skip 62 etc
        X_slice = data.iloc[idx : (idx + 48)]
        # 47 : 51, skip 62, 161:164, skip 62 etc
        y_slice = data.iloc[(idx + 48) : (idx + 48 + 4)]

        #print(f"FIRST 48 HOURS:\n{X_slice}")
        #print(f"NEXT 4 HOURS WE'RE PREDICTING:\n{y_slice}")

        X_list.append(X_slice)
        y_list.append(y_slice)

    
    X = pd.concat(X_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)

    X = X[station]
    y = y[station]
    #print(X)
    #print(y)

    # standardize 
    X_standardize = X.values.reshape(-1, 1)
    X_scaled = scaler.transform(X_standardize).flatten()

    # create a tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).view(-1, 48)

    # predict
    y_pred = model.predict(X_tensor)

    return y_pred


if __name__ == "__main__":


    # create a class for time data manipulation
    train_path = "../podatki/bicikelj_train.csv"
    timeseries_class = PrepareTimeSeries(train_path)

    # prepare data (drop nan, 48h+4h+gap etc)
    timeseries_class.load_and_process()
    # in X we have [8640 rows x 85 columns], and in y [720 rows x 85 columns]

    X = timeseries_class.X 
    y = timeseries_class.y 

    stations = timeseries_class.get_station_names()
    predictions = {}

    for idx, station in enumerate(stations):
        print(f"\n*********predicting for station number {idx+1}: {station}*********")


        # standardize data
        X_standardize = X[station].values.reshape(-1, 1)
        y_standardize = y[station].values.reshape(-1, 1)

        scaler = StandardScaler()
        scaler.fit(X_standardize)
        scaler_mean = scaler.mean_[0] 
        scaler_std = scaler.scale_[0] 

        X_standardize = scaler.transform(X_standardize).flatten()

        # create torches
        X_tensor = torch.tensor(X_standardize, dtype=torch.float32).view(-1, 48)
        y_tensor = torch.tensor(y_standardize, dtype=torch.float32).view(-1, 4) # shape (num_samples, 4)
        print(X_tensor.shape)
        # train the model
        model = Model_LSTM(scaler_mean=scaler_mean, scaler_std=scaler_std)
        model.fit(X_tensor, y_tensor)

        """
        background = X_tensor.unsqueeze(-1)  # (n, 48, 1)
        #explain_data = X_tensor[0].unsqueeze(-1)  # (50, 48, 1)
        explain_data = X_tensor.unsqueeze(-1)  # Just one sequence to explain (shape: [1, 48, 1])

        print(f"background shape: {background.shape}")
        print(f"explain_data shape: {explain_data.shape}")

        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explain_data)

        sample_idx = 0
        output_hour = 0

        # Extract SHAP values for that sample
        shap_vals = shap_values[output_hour][:, sample_idx, 0]  # shape: (48,)
        input_vals = explain_data[sample_idx].squeeze().numpy()  # shape: (48,)

        # Create the Explanation object
        explanation = shap.Explanation(
            values=shap_vals,
            data=input_vals,
            feature_names=[f"{48 - i} ure pred napovedjo" for i in range(48)]
        )

        # Plot the local bar chart
        plt.title(f"SHAP za postajo {station}")
        shap.plots.bar(explanation)

        """


        # actual test data
        test_path = "../podatki/bicikelj_test.csv"
        final_preds = final_prediction(test_path, station, model, scaler)
        predictions[station] = final_preds.numpy().flatten()  


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
        # standardize data
        timeseries_class.standardize_data(station)

        # create torches
        X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = timeseries_class.numpy_to_tensor()



        # time for NN
        mean, std = timeseries_class.get_standardization_info()
        print(f"mean: {mean}, std: {std}")

        # train the model
        model = Model_LSTM(scaler_mean=mean, scaler_std=std)
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