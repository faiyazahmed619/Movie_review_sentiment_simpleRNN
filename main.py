#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,SimpleRNN
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the word index and reverse word index

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

#load pretrained model with relu activation

model =  load_model('simpleRNN-imdb-model.h5')

#helper functions

#function to decode review

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i -3,'?') for i in encoded_review])

#function to preprocess user input

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review =sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


#streamlit app
import streamlit as st

st.title(" Review Sentiment Analysis")
st.write("Please enter a review to classify if it is positve or negative sentiment")

#user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_user_input = preprocess_text(user_input)

    #make prediction
    prediction = model.predict(preprocessed_user_input)
    sentiment = 'Positive' if prediction[0][0] > 0.4 else 'Negative'

    #display prediction
    st.write(f'Sentiment : {sentiment}')
    st.write(f'prediction score : {prediction[0][0]}')
else:
    st.write('Please enter a movie review')

    
