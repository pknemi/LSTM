import tensorflow as tf
import json

# Define vectorization parameters (use the same as in training)
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode="int", output_sequence_length=SEQUENCE_LENGTH
)

# Load your dataset again (if available)
dataset = ["Great movie!", "Horrible film.", "Amazing performance!", "Worst experience."]  # Example data

# Adapt the layer (IMPORTANT)
vectorize_layer.adapt(dataset)

# Save vocabulary manually
vocab = vectorize_layer.get_vocabulary()

with open("vocab.json", "w") as f:
    json.dump(vocab, f)

print("✅ Vocabulary saved as 'vocab.json'!")
