# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:16:55 2024

@author: seven
"""


import pandas as pd

# Read the CSV file into a DataFrame
data = pd.read_csv('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\All_Stocks_Daily_Return_All_Days_with_Z.csv')

# Group the DataFrame by Ticker
grouped_data = data.groupby('Ticker')

# Open the output file in append mode
with open('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Long_Row_Daily_Returns_1747_to_1762.csv', 'a') as f:
    # Loop through each set of 15 days in reverse order
    # leaving off 1762 since this is for training and that is what we are predicting
    for i in range(1761, 15, -1):
        # Initialize an empty DataFrame to store the row of results
        row_df = pd.DataFrame()

        # Iterate over each group
        for ticker, group in grouped_data:
            # Extract the 'Daily_Return' values for the specified range of 'Renumbered_DATE'
            daily_returns = group.loc[(group['Renumbered_DATE'] >= i - 15) & (group['Renumbered_DATE'] <= i), 'Daily_Return'].tolist()

            # Add the daily returns as a row to the DataFrame
            row_df[ticker] = daily_returns

        # Transpose the DataFrame to have Tickers as rows and 15 days as columns
        row_df = row_df.transpose()

        # Concatenate all rows into a single long row
        long_row = row_df.values.flatten()

        # Write the flattened values of the row as a single row to the CSV file
        f.write(','.join(map(str, long_row)) + '\n')





"""

import pandas as pd

# Read the CSV file into a DataFrame
data = pd.read_csv('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\All_Stocks_Daily_Return_All_Days_with_Z.csv')

# Group the DataFrame by Ticker
grouped_data = data.groupby('Ticker')

# Initialize an empty DataFrame to store the result
result_df = pd.DataFrame()

# Iterate over each group
for ticker, group in grouped_data:
    # Extract the 'Daily_Return' values for the specified range of 'Renumbered_DATE'
    daily_returns = group.loc[(group['Renumbered_DATE'] >= 1747) & (group['Renumbered_DATE'] <= 1762), 'Daily_Return'].tolist()
    
    # Add the daily returns as a row to the result DataFrame
    result_df[ticker] = daily_returns

# Transpose the result DataFrame to have Tickers as rows and 15 days as columns
result_df = result_df.transpose()

# Save the result to a CSV file
result_df.to_csv('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Daily_Returns_1747_to_1762_By_Ticker.csv', index_label='Ticker')

# Concatenate all rows into a single long row
long_row = result_df.values.flatten()

# Save the long row to a CSV file
with open('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Long_Row_Daily_Returns_1747_to_1762.csv', 'w') as f:
    # Write the flattened values as a single row to the CSV file
    f.write(','.join(map(str, long_row)))

#  loop top
# Initialize an empty DataFrame to store the second row of results
second_row_df = pd.DataFrame()



# Iterate over each group again
for ticker, group in grouped_data:
    # Extract the 'Daily_Return' values for the specified range of 'Renumbered_DATE'
    daily_returns = group.loc[(group['Renumbered_DATE'] >= 1746) & (group['Renumbered_DATE'] <= 1761), 'Daily_Return'].tolist()
    
    # Add the daily returns as a row to the second row DataFrame
    second_row_df[ticker] = daily_returns

# Transpose the second row DataFrame to have Tickers as rows and 15 days as columns
second_row_df = second_row_df.transpose()

# Concatenate all rows into a single long row
long_row2 = second_row_df.values.flatten()

# loop stop
# Save the long row to a CSV file
with open('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Long_Row_Daily_Returns_1747_to_1762.csv', 'a') as f:
    # Write a newline character to separate the first and second rows
    f.write('\n')
    
    # Write the flattened values of the second row as a single row to the CSV file
    f.write(','.join(map(str, long_row2)))

#  loop top
# Initialize an empty DataFrame to store the second row of results
third_row_df = pd.DataFrame()

# Iterate over each group again
for ticker, group in grouped_data:
    # Extract the 'Daily_Return' values for the specified range of 'Renumbered_DATE'
    daily_returns = group.loc[(group['Renumbered_DATE'] >= 1745) & (group['Renumbered_DATE'] <= 1760), 'Daily_Return'].tolist()
    
    # Add the daily returns as a row to the second row DataFrame
    third_row_df[ticker] = daily_returns

# Transpose the second row DataFrame to have Tickers as rows and 15 days as columns
third_row_df = third_row_df.transpose()

# Concatenate all rows into a single long row
long_row3 = third_row_df.values.flatten()

# loop stop
# Save the long row to a CSV file
with open('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Long_Row_Daily_Returns_1747_to_1762.csv', 'a') as f:
    # Write a newline character to separate the first and second rows
    f.write('\n')
    
    # Write the flattened values of the second row as a single row to the CSV file
    f.write(','.join(map(str, long_row3)))

#  loop top
# Initialize an empty DataFrame to store the second row of results
row4_df = pd.DataFrame()

# Iterate over each group again
for ticker, group in grouped_data:
    # Extract the 'Daily_Return' values for the specified range of 'Renumbered_DATE'
    daily_returns = group.loc[(group['Renumbered_DATE'] >= 1744) & (group['Renumbered_DATE'] <= 1759), 'Daily_Return'].tolist()
    
    # Add the daily returns as a row to the second row DataFrame
    row4_df[ticker] = daily_returns

# Transpose the second row DataFrame to have Tickers as rows and 15 days as columns
row4_df = row4_df.transpose()

# Concatenate all rows into a single long row
long_row4 = row4_df.values.flatten()

# loop stop
# Save the long row to a CSV file
with open('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Long_Row_Daily_Returns_1747_to_1762.csv', 'a') as f:
    # Write a newline character to separate the first and second rows
    f.write('\n')
    
    # Write the flattened values of the second row as a single row to the CSV file
    f.write(','.join(map(str, long_row4)))

"""



"""
# Concatenate the second row to the result DataFrame
result_df = pd.concat([result_df, second_row_df], axis=0)

# Save the updated result DataFrame to a CSV file
result_df.to_csv('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Two_Rows_Daily_Returns_1746_to_1762_By_Ticker.csv', index_label='Ticker')

# Concatenate all rows into a single long row
long_row = result_df.values.flatten()

# Save the long row to a CSV file
with open('C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Long_Row_Daily_Returns_1746_to_1762.csv', 'w') as f:
    # Write the flattened values as a single row to the CSV file
    f.write(','.join(map(str, long_row)))
"""
