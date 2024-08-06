import pandas as pd
import os
import pickle

def estimate_price(mileage, theta0, theta1):
    y_pred = theta0 + (theta1 * mileage)
    return y_pred

theta0 = 0
theta1 = 0

if os.path.exists('train_values.pkl'):
    with open('train_values.pkl', 'rb') as f:
        theta0, theta1 = pickle.load(f)
else:
    print('\033[91m' + 'ALERT: Please train the model first' + '\033[0m')

mileage = float(input('Enter the mileage of the car: '))
print('The estimated price of the car is: {:.2f}'.format(estimate_price(mileage, theta0, theta1)))


