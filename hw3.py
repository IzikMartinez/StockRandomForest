import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

"""
Closing Price
Represents Sj(t)
"""


def closing_price(stockDF, T):
    return stockDF.loc[str(T), 'Close']


def daily_rate_of_return(stockDF, T):
    today = closing_price(stockDF, T)
    yesterday = closing_price(stockDF, T - 1)
    return (today - yesterday) / yesterday


def vj(stockDF, T):
    rates_of_return = []
    for i in range(0, 15):
        rates_of_return.append(daily_rate_of_return(stockDF, T - i))
    # Invert the elements in rates_of_return
    rates_of_return = rates_of_return[:: -1]
    return rates_of_return


# Try this both ways before submitting
def true_class(targ):
    if targ >= 0.006:
        return "UP"
    elif targ <= -0.006:
        return "DOWN"
    else:
        return "STABLE"


def targ(stockDF, T):
    return daily_rate_of_return(stockDF, T+1)


def make_zt(X):
    Z = []
    for daily_rate in X.values:
        Z.append(true_class(daily_rate))
    return Z


def Xt(stocks, T):
    """
    Xt Represents an array containing all our line vector stock data "Vj(t)"
    """
    xt_list = []
    for stock, ticker in stocks:
        xt_list.append((ticker, vj(stock, T)))
    return xt_list


tickers = ["AAPL", "MSFT", "TSLA", "META", "GOOGL", "AMZN", "NVDA", "AMD", "DIS", "NFLX",
           "JPM", "KO", "BAC", "C", "WFC", "GS", "AXP", "MCD", "DJI", "SPY"]

stocks = []

for ticker in tickers:
    stocks.append((yf.Ticker(ticker).history(start='2016-01-01', end='2022-12-31'), ticker))

for stock in stocks:
    new_index = [str(i + 1) for i in range(len(stock[0]))]
    stock[0].index = new_index

############ Slide 4 ############
Y = []
for stock, ticker in stocks:
    for day in range(2, len(stock) - 1):
        Y.append((ticker, day, daily_rate_of_return(stock, day)))

# print(Y)

import matplotlib.cm as cm

colors = cm.get_cmap('rainbow', 3)  # using 'rainbow' colormap here, you can choose any
unique_tickers = list(set([x[0] for x in Y]))  # gets unique tickers


# Chunking function
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Dividing tickers into chunks of 5
chunks_of_tickers = list(chunks(unique_tickers, 3))
"""
# Now plotting
for chunk in chunks_of_tickers:
    for i, ticker in enumerate(chunk):
        ticker_data = [x for x in Y if x[0] == ticker]  # filter out data for this ticker only
        days = [x[1] for x in ticker_data]  # x-axis data
        rates_of_return = [x[2] for x in ticker_data]  # y-axis data
        plt.plot(days, rates_of_return, color=colors(i), label=ticker)
    plt.xlabel('Day')
    plt.ylabel('Rate of Return')
    plt.title('Rate of Return Over Days for Each Ticker')
    plt.legend()  # add a legend
    plt.show()
"""
############ Slide 5 ############
Xt = Xt(stocks, 17)
# print(Xt)

# Slide 6


# Slide 7

# extract vj values and convert to dataframe
X = []
for ticker, rates in Xt:
    for rate in rates:
        X.append(rate)

X_data = pd.DataFrame(X, columns=['X'])
Zt = make_zt(X_data)

# convert Zt into dataframe
Z_data = pd.DataFrame(Zt, columns=['Z'])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Create our random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)

clf.fit(X_data, Z_data.values.ravel())

oob_score = clf.oob_score_

Z_pred = clf.predict(X_data)
print(f"OOB : {oob_score}")

# Slide 8
CL0 = Z_data[Z_data['Z'] == 'STABLE']
CL1 = Z_data[Z_data['Z'] == 'UP']
CL2 = Z_data[Z_data['Z'] == 'DOWN']

s0 = CL0.shape[0]
s1 = CL1.shape[0]
s2 = CL2.shape[0]

N = Z_data.shape[0]

ratio0 = s0/N
ratio1 = s1/N
ratio2 = s2/N

print(f"Stable Ratio: {ratio0}\nUp Ratio: {ratio1}\nDown Ratio: {ratio2}")


## Increase the number of Stable and Up days
percent_increase = 0.55
percent_increase_stable = 1.00
n_up = int(len(X_data[X_data['X'] >= 0.006]) * percent_increase )
n_stable = int(len(X_data[(X_data['X'] < 0.006) & (X_data['X'] > -0.006)]) * percent_increase_stable)

print(f"Number of 'UP' entries to add {n_up}")
print(f"Number of 'STABLE' entries to add {n_stable}")
X_up = X_data[X_data['X'] >= 0.006].sample(n_up, replace=True)
X_stable = X_data[(X_data['X'] < 0.006) & (X_data['X'] > -0.006)].sample(n_stable, replace=True)
X_extended = pd.concat([X_data, X_up, X_stable])
M=X_extended.shape[0]
print(f"Size of new data set M: {M}")

Zt = make_zt(X_extended)

# convert Zt into dataframe
Z_extended_data = pd.DataFrame(Zt, columns=['Z'])
# Slide 8
CL0 = Z_extended_data[Z_extended_data['Z'] == 'STABLE']
CL1 = Z_extended_data[Z_extended_data['Z'] == 'UP']
CL2 = Z_extended_data[Z_extended_data['Z'] == 'DOWN']

s0 = CL0.shape[0]
s1 = CL1.shape[0]
s2 = CL2.shape[0]


ratio0 = s0/M
ratio1 = s1/M
ratio2 = s2/M

print(f"Altered Stable Ratio: {ratio0}\nAltered Up Ratio: {ratio1}\nAltered Down Ratio: {ratio2}")

# slide 10
import time

X_train = X_extended.drop(X_extended.index[0])
Z_train = Z_extended_data.drop(Z_extended_data.index[-1])
def TrainRandomForest(num_tree, num_select_features):
    from sklearn.ensemble import RandomForestClassifier
    # Create our random forest classifier
    random_training_size = int((2*M)/3)
    clf = RandomForestClassifier(n_estimators=num_tree, random_state=42, oob_score=True,
                                 bootstrap=True, max_samples=random_training_size,
                                 max_features=num_select_features, n_jobs=-1)
    start_time = time.time()
    # Drop the first X and last Z so that we have a list of rates of return for 299 days correlated with
    # the 299 up, down, stable labels corresponding to the next day's return
    clf.fit(X_train, Z_train.values.ravel())
    end_time = time.time()
    oob_score = clf.oob_score_
    elapsed_time = end_time - start_time
    return clf, oob_score, elapsed_time


start_time = time.time()
clf, oob_score, elapsed_time = TrainRandomForest(100, 18)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time was {elapsed_time} seconds")
predicted_z = clf.predict(X_data)
predicted_z = pd.DataFrame(predicted_z, columns=['True Class'])

from sklearn.metrics import confusion_matrix
cmatrix = confusion_matrix(Z_data, predicted_z, labels=['STABLE', 'UP', 'DOWN'])

print(cmatrix)
"""
for tx, tz in zip(X_data, test_z):
    print(f"X: {tx}, Z: {tz}")
"""


trees = [100,200,300,400,500]
selected_features = [18,36,72,140,200]
OOB_accuracies = []
computation_times = []
data = []
for tree in trees:
    for SF in selected_features:
        clf, oob_score, elapsed_time = TrainRandomForest(tree, SF)
        data.append((tree, SF, oob_score, elapsed_time))

# print(data)
df = pd.DataFrame(data, columns=['Tree', 'SF', 'oob_score', 'computation_time'])
df_pivot_oob = df.pivot(index='Tree', columns='SF', values='oob_score')
df_pivot_time = df.pivot(index='Tree', columns='SF', values='computation_time')

print(f"OOB\n {df_pivot_oob}")
print(f"Computation times (s)\n {df_pivot_time}")
# Best TR* = 100; Best SF* = 36

# Slide 13-14
"""
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# specify parameters and distributions to sample from
param_dist = {'n_estimators': [100, 200, 300, 400, 500],
              'max_features': [16, 32, 64, 'sqrt', 'log2'],
              'min_samples_split': [2, 9, 18],
              'max_leaf_nodes': [12, 48],
              'min_impurity_decrease': [0, 0.3, 0.5],
              'max_depth': [4, 6, 8],
              'criterion': ['gini', 'entropy']}

clf = RandomForestClassifier(oob_score=True, n_jobs=-1)

# run randomized search
n_iter_search = 200  # Define the number of parameter settings that are sampled.
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)

random_search.fit(X_train, Z_train.values.ravel())

# Get the best parameters and best score
print("Best parameters found by RandomizedSearchCV: ", random_search.best_params_)
print("Highest OOB score found by RandomizedSearchCV: ", random_search.best_score_)
"""

# Slide 14 end
RF_prime = RandomForestClassifier(n_estimators=500, min_samples_split=18, min_impurity_decrease=0, max_leaf_nodes=12,
                                  max_features=0.33, criterion='gini', n_jobs=-1, oob_score=True)

RF_prime.fit(X_train, Z_train)

oob_prime = RF_prime.oob_score_
importances = RF_prime.feature_importances_
print(oob_prime)
print(importances)