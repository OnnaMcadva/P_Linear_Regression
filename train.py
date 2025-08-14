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
        reader = csv.DictReader(f) # ob'ect
        for row in reader:
            print(row['mileage'], row['price'])
            mileage.append(float(row["mileage"]))
            price.append(float(row["price"]))
    return mileage, price