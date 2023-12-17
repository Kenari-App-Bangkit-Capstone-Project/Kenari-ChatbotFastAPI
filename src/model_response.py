import pandas as pd
import numpy as np
import pickle
import re
import random
import ast
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('src/mntl.csv')
model = load_model("src/baseline_model1.h5", compile=False)
tokenizer = pickle.load(open("src/tokenizer.pkl", "rb"))
X = pickle.load(open("src/maxlen.pkl", "rb"))
lbl_enc = pickle.load(open("src/lbl_enc.pkl", "rb"))

def model_response(user_input):
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', user_input)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)

    # Tokenize and pad the input sequence
    x_test = tokenizer.texts_to_sequences(text)
    x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])

    # Make predictions
    y_pred = model.predict(x_test)
    predicted_class = np.argmax(y_pred, axis=-1)
    tag = lbl_enc.inverse_transform(predicted_class)[0]

    # Get responses associated with the selected tag
    responses = df[df['tag'] == tag]['responses'].values[0]
    responses_list = ast.literal_eval(responses)

    if len(responses) > 0:
        # If there are responses, choose a random one
        response = random.choice(responses_list)
        if isinstance(response, np.ndarray):
            # Convert numpy array to string
            response = response
    else:
        response = "Maafkan saya, saya tidak mengerti hal itu."

    return tag, response