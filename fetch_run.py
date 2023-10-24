#"D:\UCLA+USC\OPT\fetch\fetch_run.py"
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import os
from collections import defaultdict
import torchvision.models as models

start_date_2021 = pd.to_datetime("2021-01-01")  # Start date for 2022
end_date_2022 = pd.to_datetime("2022-12-31")    # End date for 2022
date_range_2021_2022 = pd.date_range(start_date_2021, end_date_2022, freq='D')
x_new = pd.DataFrame({'# Date': date_range_2021_2022})

script_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_directory, 'fetch_LSTM_model.pth')

seq_length=90
input_size = seq_length
hidden_size = 64
num_layers = 2
output_size = seq_length
fetch_data_path= os.path.join(script_directory, 'data_daily.csv')


monthly_sums = defaultdict(float)
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
monthly_sum_2022 = {month: 0 for month in range(1, 13)}

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x, _ = self.lstm(input)
        x = self.linear(x)
        return x



def normalize_column(column):
  normalized = (column - column.min()) / (column.max() - column.min())
  return torch.tensor(normalized.values, dtype=torch.float32)

def revert(x,Y_min,Y_max):
  return Y_min+x*(Y_max-Y_min)


def main():

    #Load the trained LSTM model
    model = LSTM(input_size, hidden_size, output_size)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    #Load the original data
    raw = pd.read_csv(fetch_data_path)
    raw['# Date'] = pd.to_datetime(raw['# Date'])
    Y_min = raw['Receipt_Count'].values.astype(float).min()
    Y_max = raw['Receipt_Count'].values.astype(float).max()
    Y = raw['Receipt_Count'].values.astype(float)
    Y = normalize_column(raw['Receipt_Count'])
    Y = Y.reshape(-1, 1)
    Y_new = Y.detach().reshape(-1)

    #Use the loaded model to make predictions for 2022
    for j in range(365):
        with torch.no_grad():
            prediction = model(Y_new[-seq_length:].view(-1,seq_length))
        prediction = torch.tensor(prediction[0,-1].item()).view(1)
        Y_new = torch.cat((Y_new, prediction))
    output = revert(Y_new,Y_min,Y_max)

    output2 = output.detach().tolist()
    daily_number_of_receipts_2022 = output2[365:]
    start_date= 0
    for i in monthly_sum_2022.keys():
        monthly_sum = sum(daily_number_of_receipts_2022[start_date:(start_date +(days_in_month[i-1]))])
        monthly_sum_2022[i] +=monthly_sum
        start_date += days_in_month[i-1]

    x_to_be_plotted = monthly_sum_2022.keys()
    Y_to_be_plotted = [monthly_sum_2022[key] for key in monthly_sum_2022.keys() ]

    #Visualization
    plt.figure(figsize=(10, 6))
    #plt.plot(x_new['# Date'].tolist(), tensor_list, label='Predicted Number of Receipts per month', color='green', marker='o', linestyle='-')
    plt.plot(x_to_be_plotted,  Y_to_be_plotted, label='Predicted Number of Receipts per month in 2022', color='green', marker='o', linestyle='-')
    plt.xlabel('Month for 2022')
    plt.ylabel('Number of Receipts')
    plt.title('Line Plot of Monthly Number of Receipts Over Time in 2022')
    plt.legend()
    plt.grid(True)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(False)  # Disable scientific notation
    y_formatter.set_powerlimits((0, 0))  # Set the exponent range to (0, 0)
    plt.gca().yaxis.set_major_formatter(y_formatter)
    #plt.show()


    #Show the result using streamlit:

    st.title("LSTM model App for fetch analysis  By Xiaoshu Luo")
    selected_month = st.number_input("Please select a month (1-12) in 2022", min_value=1, max_value=12, step=1, value=1)
    plt.scatter(selected_month, Y_to_be_plotted[selected_month - 1], color='red', marker='o', s=100, label='Selected Month')
    st.text(f"The month you selected is: {selected_month}")
    st.text(f"The predicted monthly number of receipts in 2022 is: {int(monthly_sum_2022[selected_month])}")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
