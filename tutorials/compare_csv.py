import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_csv(file1, file2, label1, label2, output_folder = 'output_folder'):
    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Ensure the same columns in both DataFrames
    if list(df1.columns) != list(df2.columns):
        raise ValueError("Columns do not match between the two CSV files.")
    
    # Truncate df2 to match the length of df1
    max_length = len(df1) if len(df1) <= len(df2) else len(df2)
    df2 = df2.iloc[:max_length]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create plots for each column
    for column in df1.columns:
        name = column
        if '%' in column and 'Fund Rate' not in column:
            df1[column] = df1[column] * 100
            df2[column] = df2[column] * 100
        if 'USD' in column:
            name = name.replace('(USD)', '(trillion USD)')
        if '$' in column:
            name = name.replace('$', 'USD')
        plt.figure(figsize=(10, 5))
        plt.plot(df1[column], label=f'{label1} - {column}', marker='o')
        plt.plot(df2[column], label=f'{label2} - {column}', marker='x')
        plt.title(f'"{column}" from {label1} and {label2}')

        

        plt.xlabel('Days')
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        
        # Save the plot to the specified folder
        plt.savefig(f'{output_folder}/{column}_comparison.png')
        plt.show()

# Example usage
import os
ai_csv = os.getcwd() + '/simulation_results-AI-1.00_day.csv'
LabelAI = 'MARL Agents'
real_csv = os.getcwd() + '/simulation_results-real_data_day.csv'
LabelReal = 'US Government'
compare_csv(ai_csv, real_csv, LabelAI, LabelReal)
