import random
import json
import pickle
from datetime import datetime

import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
# This is now outdated, I am sure that there is a better way to do SGD
from tensorflow.python.keras.optimizer_v1 import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("~/resources/intents.json").read())

def build_and_tag_intentions(intents_json: json) -> tuple[list, list, list]:
    """
    Build initial words, classes and documents from the intents json provided as part of the resources
    :param intents_json: json loaded in statically from resources folder
    :return: tuple of the lists built from the original intents
    """
    words = []
    classes = []
    documents = []
    for intent in intents_json["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    return words, documents, classes

def lemmatize_and_sort(words: list, classes: list, letters_to_ignore=None) -> None:
    """
    Lemmatizes list of words from original intents and sorts both words and classes, then dumps back out into the pickle
    resource files
    :param words: words loaded from json file
    :param classes: classes loaded and sorted from json file
    :param letters_to_ignore: ignored letters by the model, will default to a specific list
    """
    if letters_to_ignore is None:
        letters_to_ignore = ["?", "!", ".", ","]

    words = [lemmatizer.lemmatize(word) for word in words if word not in letters_to_ignore]
    words = sorted(set(words))

    classes = sorted(set(classes))

    pickle.dump(words, open("~/resources/words.pkl", "wb"))
    pickle.dump(classes, open("~/resources/classes.pkl", "wb"))

def lemmatize_and_train(classes: list, documents: list, words: list) -> tuple[list, list]:
    """
    Lemmatize and train based on everything loaded in from the json, gives back training X and Y plots
    :param classes: classes loaded from json file
    :param documents: documents loaded from json file
    :param words: words loaded from json file
    :return: two lists as a tuple derived from the training list created
    """
    training = []

    empty_output = [0] * len(classes)
    for document in documents:
        current_collection: list = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            current_collection.append(1) if word in word_patterns else current_collection.append(0)

            output_row = list(empty_output)
            output_row[classes.index(document[1])] = 1
            training.append([current_collection, output_row])

    random.shuffle(training)
    training: np.array = np.array(training)

    return list(training[:, 0]), list(training[:, 1])

def compile_and_save_model() -> None:
    """
    Compiles, trains and saves the model into the resource file, notifies the user when training is completed.
    """
    words, documents, classes = build_and_tag_intentions(intents_json=intents)

    lemmatize_and_sort(words=words, classes=classes)

    train_x, train_y = lemmatize_and_train(words=words, classes=classes, documents=documents)

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    hist = model.fit([train_x], [train_y], epochs=200, batch_size=5, verbose=1)
    model.save("~/resources/chat_bot_model.h5", hist)
    print(f"Training completed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")


if __name__ == '__main__':
    # Will also train the model
    compile_and_save_model()
