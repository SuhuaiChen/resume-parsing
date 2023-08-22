import pickle
import re
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter

"""
dataset source: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
colab notebook: https://colab.research.google.com/drive/1nzEQ-M6F6n_0xSzlF4VGO5-xIVT8p6Rd#scrollTo=c8wG6Gc_dd3a
"""


def df2lists(df):
    records = df.values.tolist()
    texts = []
    labels = []
    for record in records:
        text = []
        raw = record[2]
        label = record[3]
        soup = BeautifulSoup(raw, 'lxml')
        descriptions = soup.find_all('span', 'jobline')
        for description in descriptions:
            lis = description.find_all('li')
            for li in lis:
                li_text = li.get_text()
                li_text = re.sub('\n', '', li_text).strip()
                if li_text:
                    text.append(li_text)
        texts.append(text)
        labels.append(label)
    return [texts, labels]


def create():
    df_all = pd.read_csv('Resume.csv')
    df_all = df_all.sample(frac=1)

    tenth = int(len(df_all)/10)
    df_train = df_all.iloc[:tenth*8]
    df_valid = df_all.iloc[tenth*8:tenth*9]
    df_test = df_all.iloc[tenth*9:]

    train = df2lists(df_train)
    valid = df2lists(df_valid)
    test = df2lists(df_test)

    train_texts = []
    train_labels = []
    for text, label in zip(train[0], train[1]):
        train_texts.extend(text)
        train_labels.extend([label]*len(text))
    print(len(train_texts))
    print(len(train_labels))

    train = [train_texts, train_labels]
    with open("train_new.pickle", 'wb') as f:
        pickle.dump(train, f)

    with open("valid_new.pickle", 'wb') as f:
        pickle.dump(valid, f)

    with open("test_new.pickle", 'wb') as f:
        pickle.dump(test, f)


def distribution():
    with open('train_new.pickle', 'rb') as f:
        data = pickle.load(f)
    labels = data[1]
    label_count = Counter(labels)
    print(label_count)





