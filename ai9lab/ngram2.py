# Import necessary libraries
import nltk
from nltk import trigrams
from nltk.corpus import reuters
from collections import defaultdict

# Download necessary NLTK resources
nltk.download('reuters')
nltk.download('punkt')

# Tokenize the text
words = [word.lower() for word in nltk.word_tokenize(' '.join(reuters.words()))]  # <-- notice .lower()

# Create trigrams
tri_grams = list(trigrams(words))

# Build a trigram model
model = defaultdict(lambda: defaultdict(lambda: 0))

# Count frequency of co-occurrence
for w1, w2, w3 in tri_grams:
    model[(w1, w2)][w3] += 1

# Transform the counts into probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

# Function to predict the next word
def predict_next_word(w1, w2):
    """
    Predicts the next word based on the previous two words using the trained trigram model.
    Args:
    w1 (str): The first word.
    w2 (str): The second word.

    Returns:
    str: The predicted next word.
    """
    w1, w2 = w1.lower(), w2.lower()  # <-- lowercase input words
    next_words = model.get((w1, w2), None)
    if next_words and sum(next_words.values()) > 0:
        predicted_word = max(next_words, key=next_words.get)
        return predicted_word
    else:
        return "No prediction available"

# Example usage
print("Next Word:", predict_next_word('the', 'stock'))
print("Next Word:", predict_next_word('new', 'york'))
print("Next Word:", predict_next_word('interest', 'rates'))
print("Next Word:", predict_next_word('according', 'to'))



