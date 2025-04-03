import streamlit as st
import tensorflow as tf
import numpy as np
import json

# Load the trained model
model = tf.keras.models.load_model("sentiment_model.h5")

# Define the vectorization layer
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode="int", output_sequence_length=SEQUENCE_LENGTH
)

# Load vocabulary from vocab.json
try:
    with open("vocab.json", "r") as f:
        vocab = json.load(f)
    vectorize_layer.set_vocabulary(vocab)  # Set vocabulary
except FileNotFoundError:
    st.error("⚠ Vocabulary file 'vocab.json' not found! Run the vocabulary-saving script first.")

# Streamlit UI with columns
col1, col2 = st.columns([1, 2])  # Adjust column width ratios

with col1:
    st.image("empuraan.jpg", width=300)  # Adjusted width & height

with col2:
    st.markdown("### Empuraan Sentiment Analyzer 🎬")
    st.write("Type your review and see if you're hyped or disappointed! 🚀🎥")
    user_review = st.text_area("Your Review:", height=100)  # Reduced height

if st.button("🔥 Analyze My Review 🔍"):
    if user_review:
        input_text = tf.constant([user_review])
        vectorized_text = vectorize_layer(input_text)
        prediction = model.predict(vectorized_text)[0][0]  # Assuming sigmoid output

        if prediction > 0.5:
            st.success("🔥 Empuraan is a blockbuster in your books! Loved the hype! 🔥")
        else:
            st.error("😕 Not impressed? Empuraan didn't live up to your expectations.")
    else:
        st.warning("⚠ Enter a review before clicking!")
