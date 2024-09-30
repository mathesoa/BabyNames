import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NameSimilarity:
    def __init__(self, db_path='baby_names.db'):
        # Initialize the SQLite connection
        self.conn = sqlite3.connect(db_path)

    def load_data(self, df):
        """ Load a pandas DataFrame into the SQLite database """
        df.to_sql('baby_names', self.conn, if_exists='replace', index=False)
        # Create an index on the first_letter column for faster queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_first_letter ON baby_names (first_letter);")

    def query_names_by_letter(self, letter):
        """ Query the database for names starting with a given letter """
        query = f"SELECT Name FROM baby_names WHERE first_letter = '{letter}'"
        return pd.read_sql_query(query, self.conn)["Name"].tolist()

    def find_similar_names(self, name, names_list, top_n=5):
        """ Find the most similar names to a given name using TF-IDF and Cosine Similarity """
        # Vectorize the names using character-level n-grams
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        tfidf_matrix = vectorizer.fit_transform(names_list)

        # Compute cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix)

        # Find the index of the given name
        if name not in names_list:
            return f"{name} not found in list."
        idx = names_list.index(name)

        # Get similarity scores for the name
        sim_scores = list(enumerate(cosine_similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Return the top N similar names (excluding the name itself)
        return [(names_list[i], score) for i, score in sim_scores[1:top_n+1]]

    def get_similar_names_for_letter(self, name, letter, top_n=5):
        """ Get similar names for a specific name in the dataset of names starting with a given letter """
        # Query names starting with the given letter
        names_list = self.query_names_by_letter(letter)

        # Find similar names
        return self.find_similar_names(name, names_list, top_n)
