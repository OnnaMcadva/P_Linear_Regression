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
    normalized = []