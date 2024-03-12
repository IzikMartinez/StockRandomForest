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
    if targ > 0.006:
        return "UP"
    elif targ < 0.006:
        return "DOWN"
    else:
        return "STABLE"


def targ(stockDF, T):
    return daily_rate_of_return(stockDF, T+1)


def Zt(stockDF):
    Z = []
    for i in range(1, len(stockDF)-1):
        Z.append(true_class(targ(stockDF, i)))
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
Zt = Zt(stocks[19][0])

# Slide 7

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# extract vj values and convert to dataframe
X_data = pd.DataFrame([tpl[1] for tpl in Xt], columns=[f'Rate_{i}' for i in range(1, 16)])
X_data = X_data.values.reshape(-1,1)
# convert Zt into dataframe
Z_data = pd.DataFrame(Zt, columns=['Z'])

# Create our random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_data, Z_data)

"""
Z_pred = clf.predict(X_test)
accuracy = accuracy_score(Z_test, Z_pred)
print(f"Accuracy: {accuracy}")
"""

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

print(f"Ratio0: {ratio0}, Ratio1: {ratio1}, Ratio2: {ratio2}")