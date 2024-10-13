import json
import pickle
import nltk

import random as r
import numpy as np

from typing import Any
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model

model = load_model("~/resources/chat_bot_model.h5")

lemmatizer = WordNetLemmatizer()

def load_static_resources() -> tuple[json, pickle, pickle]:
    """
    Load in all static resources and return them as a tuple for later usage
    :return: tuple of all static resources
    """
    intents = json.loads(open("~/resources/intents.json").read())
    words = pickle.load(open("~/resources/words.pkl", "rb"))
    classes = pickle.load(open('~/resources/classes.pkl', "rb"))

    return intents, words, classes

def sanitize_user_input(user_input: str) -> list[str]:
    """
    Sanitize and tokenize a given user input
    :param user_input: usually handled from command line
    :return: tokenized input after it has been run through nltk
    """
    tokenized_input: list = nltk.word_tokenize(user_input)
    tokenized_input = [lemmatizer.lemmatize(word) for word in tokenized_input]
    return tokenized_input

def get_collection_of_words(user_input: str) -> np.array:
    """
    Returns collection of words and assigns weights to the collections
    :param user_input: usually handled from command line
    :return: numpy array with weights and words
    """
    _, words, _ = load_static_resources()
    sanitized_input: list[str] = sanitize_user_input(user_input=user_input)
    collection: list[int] = [0] * len(sanitized_input)
    for w in sanitized_input:
        for i, word in enumerate(words):
            if word == w:
                collection[i] = 1

    return np.array(collection)

def predict_class(user_input: str, error_threshold: int = 0.25) -> list[dict[str, str | Any]]:
    """
    Predicts the likelihood that the intent to be returned in the best option to return to the user
    :param user_input: original input
    :param error_threshold: defaults to 0.25 but provides options for it ot be changed
    :return: List of dictionaries, that include intents and probability the model thinks thats the correct answer
    """
    _, _, classes = load_static_resources()
    word_collection: np.array = get_collection_of_words(user_input=user_input)
    result = model.predict(np.array([word_collection]))[0]
    results = sorted([[i, r] for i, r in enumerate(result) if r > error_threshold], key=lambda x: x[1], reverse=True)
    list_to_return = []
    for r in results:
         list_to_return.append({"intent": classes[r[0]], "probability": str(r[1])})
    return list_to_return

def generate_response(list_of_intents: list[dict[str, str | Any]]) -> str:
    """
    Generate a response based on the static intents file
    :param list_of_intents: provided list of intents generated from earlier predictions
    :return: response as string to the user
    """
    tag: str = list_of_intents[0]["intent"]
    _, _, intents = load_static_resources()
    intents_list = intents["intents"]
    for i in intents_list:
        if i["tag"] == tag:
            result = r.choice(i["responses"])
            break
    return result