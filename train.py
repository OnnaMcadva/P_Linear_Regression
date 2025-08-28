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
    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            if not {"km", "price"}.issubset(reader.fieldnames):
                raise ValueError(
                    "File data.csv must contain columns 'km' and 'price'.")

            for row in reader:
                try:
                    mileage.append(float(row["km"]))
                    price.append(float(row["price"]))
                except (ValueError, KeyError) as e:
                    print(f"Skipped row with invalid data: {row}. Error: {e}")
                    continue

        if not mileage or not price:
            raise ValueError("File *.csv is empty or contains no valid data.")

        if len(mileage) < 2:
            raise ValueError("Insufficient data in *.csv for model training.")

        if any(p <= 0 for p in price):
            raise ValueError("Prices in *.csv file must be positive.")

        if any(m < 0 for m in mileage):
            raise ValueError("Mileage in *.csv file cannot be negative.")

        return mileage, price

    except FileNotFoundError:
        raise FileNotFoundError("File *.csv not found.")
    except Exception as e:
        raise ValueError(f"Error reading *.csv file: {str(e)}")


def normalize_feature(values):
    mean_val = sum(values) / len(values)
    range_val = max(values) - min(values)
    if range_val == 0:
        normalized = [0.0 for _ in values]
        range_val = 1.0
    else:
        normalized = [(v - mean_val) / range_val for v in values]
    return normalized, mean_val, range_val


def normalize(value, mean, range_val):
    return 0.0 if range_val == 0 else (value - mean) / range_val


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
    if range_mileage != 0:
        theta1_raw = theta1 / range_mileage
        theta0_raw = theta0 - theta1 * mean_mileage / range_mileage
    else:
        theta1_raw = 0.0
        theta0_raw = theta0

    with open(filename, "w") as f:
        json.dump({
            "theta0": theta0,
            "theta1": theta1,
            "mean_mileage": mean_mileage,
            "range_mileage": range_mileage,
            "theta0_raw": theta0_raw,
            "theta1_raw": theta1_raw
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

    theta0, theta1 = train(
        normalized_mileage,
        price,
        LEARNING_RATE,
        ITERATIONS
    )

    save_model(theta0, theta1, mean_m, range_m, THETA_FILE)

    print(f"Training completed: theta0 = {theta0}, theta1 = {theta1}")
    print(
        f"Model R² score: "
        f"{calculate_r2(mileage, price, theta0, theta1, mean_m, range_m):.4f}"
    )

    plot_data_and_regression(mileage, price, theta0, theta1, mean_m, range_m)
