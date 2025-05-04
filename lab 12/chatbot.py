import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class UniversityQnABot:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.data["question"])

    def get_response(self, user_input):
        user_vec = self.vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, self.X)
        max_sim = similarity.max()

        if max_sim > 0.4:
            index = similarity.argmax()
            return self.data["answer"].iloc[index]
        else:
            return "Sorry, I don't have an answer for that."
