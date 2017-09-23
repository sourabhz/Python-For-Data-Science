import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import sys


''' plt.switch_backend(new_backend) if it doesn't plot on your machine'''

# two new empty lists

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))  # removing slashes with split
            prices.append(float(row[1]))
    return  # to finish our with block


def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel= 'linear', C= 1e3)  # 1e3 scientific notation for 1000

    '''in math forklore the no free lunch theorem states that there are no gurantees
    for one optimization to work better than the other, so we will try both'''
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree=2)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)

    # lets fit or train our model in our dates and price data

    svr_rbf.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_lin.fit(dates, prices)

    plt.scatter(dates, prices, color = 'black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label = 'RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label = 'Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label = 'Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('aapl.csv')

predicted_price = predict_prices(dates,prices, 15)

print(predicted_price)






