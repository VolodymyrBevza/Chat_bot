import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLeammatizer

from tensfwlow.keras.models import Sequential
from tensfwlow.keras.layers import Dense, Activation, Dropout
from tensfwlow.keras.optimizers import SGD

lemmatizer = WordNetLeammatizer()

tasks = json.loads(open('tasks.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for task in tasks['task']:
    for pattern in tasks:
        word_lsit = nltk.word_tokenize(pattern)
        words.append(word_lsit)
        documents.append((word_lsit, task['tag']))
        if task['tag'] not in classes:
            classes.append(task['tag'])

words = [lemmatizer.lematize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open(words.pkl, 'wb'))
pickle.dump(classes, open(classes.pkl, 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemetize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_raw = list(output_empty)
    output_raw[classes.index(document[1])]
    training.append([bag, output_raw])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimazer=sgd, mettrics='accuracy')

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

