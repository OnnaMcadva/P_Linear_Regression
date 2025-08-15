#!/usr/bin/env python3
import csv
import json

DATA_FILE = "data.csv"
THETA_FILE = "thetas.json"

LEARNING_RATE = 0.1
ITERATIONS = 1000

def load_data(filename):
    mileage = []
    price = []
    with open(filename, "r") as f: # "r" reading only / with гарантирует корректное закрытие файла
        reader = csv.DictReader(f) # *helpers.txt
        for row in reader:
            print(row['km'], row['price'])
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

def save_model()