import os
import csv
import re

import pandas as pd

from app.preprocessing import preprocessing

folder_path = "train"

files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

with open('train.csv', mode='w', encoding='utf-8', newline='') as output_file:
    csv_writer = csv.writer(output_file, delimiter=';')
    csv_writer.writerow(['Text', 'Score'])
    for file_name in files:
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            text = file.read()
            text = re.sub(',', '', text)
            text = re.sub(';', '', text)
        last_digit = file_name[-6:-4]
        last_digit = re.sub('_', '', last_digit)
        csv_writer.writerow([text, last_digit])

train = pd.read_csv('train.csv', delimiter=';', quotechar='"')
train['Text'] = train['Text'].apply(preprocessing)
train.to_csv('train_prep.csv', index=False)

folder_path = "test"

files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

with open('test.csv', mode='w', encoding='utf-8', newline='') as output_file:
    csv_writer = csv.writer(output_file, delimiter=';')
    csv_writer.writerow(['Text', 'Score'])
    for file_name in files:
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            text = file.read()
            text = re.sub(',', '', text)
            text = re.sub(';', '', text)
        last_digit = file_name[-6:-4]
        last_digit = re.sub('_', '', last_digit)
        csv_writer.writerow([text, last_digit])

test = pd.read_csv('test.csv', delimiter=';', quotechar='"')
test['Text'] = test['Text'].apply(preprocessing)
test.to_csv('test_prep.csv', index=False)
