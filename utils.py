import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy


# Evaluation
def evaluate_model(model, data):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(data)):
            input_data = torch.tensor(data['standardized_input'].iloc[i]).unsqueeze(0).float()
            output_data = torch.tensor(data['standardized_output'].iloc[i]).float()
            output = model(input_data)
            _, predicted = torch.max(output.data, 1)
            total += output_data.size(0)
            correct += (predicted == output_data).sum().item()
    print(f"Accuracy: {correct / total}")

def predict(model, data):
    predictions = []
    with torch.no_grad():
        for i in range(len(data)):
            input_data = torch.tensor(data['standardized_input'].iloc[i]).float()
            output = model(input_data)
            predictions.append(output)
    return predictions

def standardize_data(data):
    temp_data = copy.deepcopy(data)
    for i in range(len(temp_data)):
        for j in range(len(temp_data[i])):
            while len(temp_data[i][j]) < 30:
                temp_data[i][j].append(-1)
        while len(temp_data[i]) < 30:
            temp_data[i].append([-1 for k in range(30)])
    return temp_data


def filter_data_by_id(id, dataframe, solution=True):
    """
    Filter data by id
    :param id: id of the data
    :param dataframe: dataframe to filter
    :param solution: whether to return solution or not
        
    :return: test_data_input, test_data_solution, train_data_inputs, train_data_outputs
    """
    
    data = dataframe[dataframe['id'] == id]

    test_data = data['test']
    test_data_input = test_data.iloc[0][0]['input']
    if solution:
        test_data_solution = test_data.iloc[0][0]['output']

    train_data = data['train']
    train_data_inputs = [i['input'] for i in train_data.iloc[0]]
    train_data_outputs = [i['output'] for i in train_data.iloc[0]]

    test_data = data['test']
    test_data_input = test_data.iloc[0][0]['input']

    if solution:
        test_data_solution = test_data.iloc[0][0]['output']
        return test_data_input, test_data_solution, train_data_inputs, train_data_outputs
    return test_data_input, train_data_inputs, train_data_outputs

# Define function to visualize each data
colors = ["White", "Black", "Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange"]

def visualize_from_data(input_data, output_data=None):
    """
    Visualize input and output data
    :param input_data: input data needs to be formatted as a list of 2D arrays
    :param output_data: output data

    :return: None
    """

    num_inputs = len(input_data)
    has_output = output_data is not None

    # If we have output data, we want 2 rows; otherwise, just 1 row
    num_rows = 2 if has_output else 1
    num_cols = num_inputs

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # If there is only one column, axes is not a 2D array; make it so
    if num_cols == 1:
        if has_output:
            axes = [[axes[0]], [axes[1]]]
        else:
            axes = [axes]

    # If there is only one row, make sure axes is a 2D array
    if num_rows == 1:
        axes = [axes]

    for i, data in enumerate(input_data):
        ax = axes[0][i]
        ax.imshow(data, cmap='tab20', norm=plt.Normalize(0, 19))
        ax.set_title(f"Input {i + 1}")
        ax.axis('off')

    if has_output:
        for i, data in enumerate(output_data):
            ax = axes[1][i]
            ax.imshow(data, cmap='tab20', norm=plt.Normalize(0, 19))
            ax.set_title(f"Output {i + 1}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_from_id(id, dataframe, solution=True):
    _, _, input_data, output_data = filter_data_by_id(id, dataframe, solution)
    visualize_from_data(input_data, output_data)

# Function to convert JSON data to DataFrame
def json_to_dataframe(challenges_data, solutions_data=None):
    """
    Convert JSON data to a DataFrame.
    :param challenges_data: JSON data containing challenges.
    :param solutions_data: JSON data containing solutions.

    :return: DataFrame containing challenges and solutions.
    """
    records = []
    for key, value in challenges_data.items():
        record = {
            'id': key,
            'train': [
                {
                    'input': item['input'],
                    'output': item['output']
                }
                for item in value['train']
            ] if 'train' in value else [],
            'test': [
                {
                    'input': item['input'],
                }
                for item in value['test']
            ] if 'test' in value else []
        }
        if solutions_data and key in solutions_data:
            for i, item in enumerate(record['test']):
                item['output'] = solutions_data[key][i] if i < len(solutions_data[key]) else None
        records.append(record)
    return pd.DataFrame(records)