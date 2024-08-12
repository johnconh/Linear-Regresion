import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('data.csv')

x = data['km']
y = data['price']

x_mean = np.mean(x)
x_std = np.std(x)
y_mean = np.mean(y)
y_std = np.std(y)

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def denormalize(norm_data, original_mean, original_std):
    return norm_data * original_std + original_mean

x = normalize(x)
y = normalize(y)

theta0 = 0
theta1 = 0

learning_rate = 0.01
iterations = 1001
m = len(y)

def estimate_price(mileage, theta0, theta1):
    y_pred = theta0 + (theta1 * mileage)
    return y_pred

def Mean_Squared_Error(y, y_pred):
    return np.sum((y - y_pred) ** 2) / len(y)
    
mse_history = []

for i in range(iterations):
    
    error = estimate_price(x, theta0, theta1) - y
    theta0 = theta0 - learning_rate * (1/m) * np.sum(error)
    theta1 = theta1 - learning_rate * (1/m) * np.sum(error * x)
    
    mse = Mean_Squared_Error(y, estimate_price(x, theta0, theta1))
    mse_history.append(mse)

    if i % 20 == 0:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1,2,1)
        plt.scatter(x, y, color='blue', label='Data points')
        plt.plot(x, estimate_price(x, theta0, theta1), color='red', label='Regression line')
        plt.xlabel('Mileage (km)')
        plt.ylabel('Price ($)')
        plt.title('Mileage vs Price - Iteration {}'.format(i))
        plt.legend()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.plot(range(i+1), mse_history[:i+1], label='MSE')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error Progression')
        plt.legend()
        plt.xticks([])
        plt.yticks([])

        plt.savefig('training_progresion.png')      
        plt.close()

x = denormalize(x, x_mean, x_std)
y = denormalize(y, y_mean, y_std)
theta0 = theta0 * y_std + y_mean - theta1 * x_mean * y_std / x_std
theta1 = theta1 * y_std / x_std
mean_abs_error = np.mean(np.abs(y -(theta0 + (theta1*x))))

with open('train_values.pkl', 'wb') as f:
    pickle.dump((theta0, theta1), f)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, estimate_price(x, theta0, theta1), color='red', label='Regression line')
plt.xlabel('Mileage (km)')
plt.ylabel('Price ($)')
plt.title('Mileage vs Price')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(iterations), mse_history, label='MSE')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Mean Squared Error Progression')
plt.legend()

plt.figtext(0.5, 0.97, 'Learning rate: {}\nNumber of cars: {}'.format(learning_rate, m), ha='center', va='top')

plt.savefig('progresion.png')
plt.close()


print('\nMean Squared Error:', mse)
print('\nMean Abs error:', mean_abs_error)
print('\033[92m' + 'Model trained and saved.' + '\033[0m\n')
