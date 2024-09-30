import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer and model based on user selection
def load_model_and_tokenizer(selected_model):
    if selected_model == 'Gender Neutral':
        model = tf.keras.models.load_model('gender_neutral_name_generation_model.keras')
        with open('gender_neutral_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    elif selected_model == 'Boy':
        model = tf.keras.models.load_model('boy_name_generation_model.keras')
        with open('boy_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    elif selected_model == 'Girl':
        model = tf.keras.models.load_model('girl_name_generation_model.keras')
        with open('girl_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    
    return model, tokenizer

# Function to generate a name

def generate_name(model, tokenizer, seed_text, next_chars):
    for _ in range(next_chars):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Check if token_list is empty
        if not token_list:
            st.error("Error: Unrecognized characters in the seed text. Please try a different input.")
            return seed_text  # Return the current seed_text and stop generation

        token_list = pad_sequences([token_list], maxlen=10, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = tf.argmax(predicted_probs, axis=-1).numpy()[0]

        output_char = ""
        for char, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_char = char
                break

        seed_text += output_char

    return seed_text