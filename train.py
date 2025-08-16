#!/usr/bin/env python3
import csv
import json
import matplotlib.pyplot as plt

DATA_FILE = "data.csv"
THETA_FILE = "thetas.json"

LEARNING_RATE = 0.1
ITERATIONS = 1000

def load_data(filename):
    mileage = []
    price = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mileage.append(float(row["km"]))
            price.append(float(row["price"]))
    return mileage, price

def normalize_feature(values):
    mean_val = sum(values) / len(values)
    range_val = max(values) - min(values)
    normalized = [(v - mean_val) / range_val for v in values]
    return normalized, mean_val, range_val

def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

def train(mileage, price, learning_rate, iterations):
    m = len(mileage)
    theta0 = 0.0
    theta1 = 0.0

    for _ in range(iterations):
        sum_error_theta0 = 0.0
        sum_error_theta1 = 0.0

        for i in range(m):
            prediction = estimate_price(mileage[i], theta0, theta1)
            error = prediction - price[i]
            sum_error_theta0 += error
            sum_error_theta1 += error * mileage[i]

        tmp_theta0 = theta0 - (learning_rate * (sum_error_theta0 / m))
        tmp_theta1 = theta1 - (learning_rate * (sum_error_theta1 / m))

        theta0, theta1 = tmp_theta0, tmp_theta1

    return theta0, theta1

def save_model(theta0, theta1, mean_mileage, range_mileage, filename):
    with open(filename, "w") as f:
        json.dump({
            "theta0": theta0,
            "theta1": theta1,
            "mean_mileage": mean_mileage,
            "range_mileage": range_mileage
        }, f)

def plot_data_and_regression(mileage, price, theta0, theta1, mean_m, range_m):
    
    plt.scatter(mileage, price, color="blue", label="Data")

    min_m, max_m = min(mileage), max(mileage)
    line_x = [min_m, max_m]
    norm_x = [(x - mean_m) / range_m for x in line_x]
    line_y = [estimate_price(xn, theta0, theta1) for xn in norm_x]
    plt.plot(line_x, line_y, color="red", label="Regression line")

    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Car Price Prediction")
    plt.legend()
    plt.show()

def calculate_r2(mileage, price, theta0, theta1, mean_m, range_m):
    predictions = [
        estimate_price((x - mean_m) / range_m, theta0, theta1)
        for x in mileage
    ]
    mean_price = sum(price) / len(price)
    ss_total = sum((y - mean_price) ** 2 for y in price)
    ss_residual = sum((y - p) ** 2 for y, p in zip(price, predictions))
    r2 = 1 - (ss_residual / ss_total)
    return r2

if __name__ == "__main__":
    mileage, price = load_data(DATA_FILE)

    normalized_mileage, mean_m, range_m = normalize_feature(mileage)

    theta0, theta1 = train(normalized_mileage, price, LEARNING_RATE, ITERATIONS)

    save_model(theta0, theta1, mean_m, range_m, THETA_FILE)

    print(f"Training completed: theta0 = {theta0}, theta1 = {theta1}")
    print(f"Model R² score: {calculate_r2(mileage, price, theta0, theta1, mean_m, range_m):.4f}")

    plot_data_and_regression(mileage, price, theta0, theta1, mean_m, range_m)