#!/usr/bin/env python3
import json
import csv
import matplotlib.pyplot as plt

DATA_FILE = "data.csv"
THETA_FILE = "thetas.json"

def load_model(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return (
                data["theta0"],
                data["theta1"],
                data["mean_mileage"],
                data["range_mileage"]
            )
    except FileNotFoundError:
        print("Error: thetas.json not found. Please train the model first.")
        exit(1)

def load_data(filename):
    mileage = []
    price = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mileage.append(float(row["km"]))
            price.append(float(row["price"]))
    return mileage, price

def normalize(value, mean, range_val):
    return (value - mean) / range_val

def estimate_price(mileage, theta0, theta1, mean_m, range_m):
    normalized = normalize(mileage, mean_m, range_m)
    return theta0 + theta1 * normalized

def plot_prediction(mileage, price, theta0, theta1, mean_m, range_m, input_mileage, predicted_price):

    plt.scatter(mileage, price, color="blue", label="Data")

    min_m, max_m = min(mileage), max(mileage)
    line_x = [min_m, max_m]
    norm_x = [(x - mean_m) / range_m for x in line_x]
    line_y = [theta0 + theta1 * nx for nx in norm_x]
    plt.plot(line_x, line_y, color="red", label="Regression line")

    # oracleOfDelphi ^-^
    plt.scatter([input_mileage], [predicted_price], color="green", s=100, zorder=5, label="Prediction")

    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Car Price Prediction")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    theta0, theta1, mean_m, range_m = load_model(THETA_FILE)

    try:
        mileage_input = float(input("Enter mileage: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        exit(1)

    predicted_price = estimate_price(mileage_input, theta0, theta1, mean_m, range_m)
    print(f"Estimated price: {predicted_price}")

    # Graph
    mileage, price = load_data(DATA_FILE)
    plot_prediction(mileage, price, theta0, theta1, mean_m, range_m, mileage_input, predicted_price)