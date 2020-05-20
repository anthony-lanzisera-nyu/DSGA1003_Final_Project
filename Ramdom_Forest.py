import csv
import string
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def create_input(filename):
    text = []
    rating = []
    label = []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if "\n" not in row[6]: # remove malformed example
                text.append(row[6].lower())
                rating.append(row[3])
                label.append(row[4])
    return [text[1:], rating[1:], label[1:]]


def text_to_token(text):
    translator = str.maketrans('', '', string.punctuation)
    data = []
    for line in text:
        line = line.translate(translator)
        tokens = []
        for word in line.split(" "):
            tokens.append(word)
        data.append(tokens)
    return data


def select_n_feature(data, n):
    wordDict = {}
    for line in data:
        for word in line:
            if word in wordDict:
                wordDict[word] += 1
            else:
                wordDict[word] = 1

    sorted_wordDict = {k: v for (k, v) in sorted(wordDict.items(), key=lambda item: item[1], reverse=True)}
    return [k for k in list(sorted_wordDict)[100:n+100]]


def transform(text, rating, feature):
    matrix = np.zeros((len(text), len(feature)))

    for i in range(len(text)):
        line = text[i]
        wordDict = Counter(line)
        for j in range(len(feature)):
            if j == 0:
                matrix[i][j] = rating[i]
            if feature[j] in line:
                matrix[i][j] = wordDict[feature[j]]
    return matrix


# create train, evaluation dataset
train_text, train_rating, train_label = create_input('train.csv')
dev_text, dev_rating, dev_label = create_input('dev.csv')

# tokenize sentence
train_text = text_to_token(train_text)
dev_text = text_to_token(dev_text)

# select top n features
feature = select_n_feature(train_text, 2048)

# create data matrix
train_text = transform(train_text, train_rating, feature)
dev_text = transform(dev_text, train_rating, feature)

# Hyperparameter search for the model
params = {
    'max_depth': [None, 10, 30, 50, 70, 90, 100],
    'min_samples_split': [2, 3, 5, 7, 9, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': [None, 'sqrt', 'log2'],
    'n_estimators': [100, 200, 400, 600, 800, 1000, 1400, 1700, 2000],
    'sampling_strategy': [0.3, 0.5, 0.7, 0.9, 1],
    'class_weight': ['balanced', {'0': 10, '1': 1}, {'0': 30, '1': 1}, {'0': 50, '1': 1}]
}

from imblearn.over_sampling import SMOTE

ovr = SMOTE(sampling_strategy=0.5, random_state = 42)

train_text_one, train_label_one = ovr.fit_resample(train_text[:50000], train_label[:50000])
train_text_two, train_label_two = ovr.fit_resample(train_text[50000:100000], train_label[50000:100000])
train_text_three, train_label_three = ovr.fit_resample(train_text[100000:150000], train_label[100000:150000])
train_text_four, train_label_four = ovr.fit_resample(train_text[150000:200000], train_label[150000:200000])
train_text_five, train_label_five = ovr.fit_resample(train_text[200000:], train_label[200000:])

train_resample_text = np.concatenate((train_text_one, train_text_two, train_text_three, train_text_four, train_text_five))
train_resample_label = np.concatenate((train_label_one, train_label_two, train_label_three, train_label_four, train_label_five))

# baseline
rf = RandomForestClassifier(random_state=0)
rf.fit(train_text, train_label)
pred = rf.predict_proba(dev_text)
print("AUROC:", roc_auc_score(dev_label, pred[:, 1]))
print("AP:", average_precision_score(dev_label, pred[:, 1], pos_label="1"))

# baseline with oversampling
rf = RandomForestClassifier(random_state=0)
rf.fit(train_resample_text, train_resample_label)
pred = rf.predict_proba(dev_text)
print("AUROC:", roc_auc_score(dev_label, pred[:, 1]))
print("AP:", average_precision_score(dev_label, pred[:, 1], pos_label="1"))

# hyperparameter tuning
cv = RandomForestClassifier(random_state=0, n_estimators=200, max_depth=90, min_samples_split=5, min_samples_leaf=5, max_features='log2', class_weight='balanced')
cv.fit(train_text, train_label)
pred = cv.predict_proba(dev_text)
print("Original Dev auROC:", roc_auc_score(dev_label, pred[:, 1]))
print("Original Dev AP:", average_precision_score(dev_label, pred[:, 1], pos_label="1"))_

# hyperparameter tuning with oversampling
cv = RandomForestClassifier(random_state=0, n_estimators=200, max_depth=90, min_samples_split=5, min_samples_leaf=5, max_features='log2', class_weight='balanced')
cv.fit(train_resample_text, train_resample_label)
pred = cv.predict_proba(dev_text)
print("Original Dev auROC:", roc_auc_score(dev_label, pred[:, 1]))
print("Original Dev AP:", average_precision_score(dev_label, pred[:, 1], pos_label="1"))_

