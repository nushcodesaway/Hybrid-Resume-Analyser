from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pandas as pd


# 🔥 Load semantic model (only once)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


# 🔹 1. Lexical Similarity (TF-IDF)
def get_lexical_similarity(docs, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs + [query])
    
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return similarity[0]


# 🔹 2. Semantic Similarity (BERT Embeddings)
def get_semantic_similarity(docs, query):
    doc_embeddings = semantic_model.encode(docs)
    query_embedding = semantic_model.encode([query])
    
    similarity = cosine_similarity(query_embedding, doc_embeddings)
    return similarity[0]


# 🔹 3. Hybrid Similarity (BEST PART 🔥)
def get_hybrid_similarity(docs, query):
    lexical_scores = get_lexical_similarity(docs, query)
    semantic_scores = get_semantic_similarity(docs, query)
    
    # Combine both
    final_scores = 0.5 * lexical_scores + 0.5 * semantic_scores
    return final_scores


# 🔹 4. Anomaly Detection (IoT behavior)
def detect_anomalies(behavior_list):
    df = pd.DataFrame(behavior_list)
    
    model = IsolationForest(random_state=42)
    df['anomaly'] = model.fit_predict(df)
    
    return df


# 🔹 5. Clustering (Grouping candidates)
def cluster_data(similarity_scores):
    n = len(similarity_scores)
    
    # Ensure clusters <= number of samples
    k = min(3, n)

    if k == 0:
        return []

    model = KMeans(n_clusters=k, random_state=42)

    data = [[s] for s in similarity_scores]
    labels = model.fit_predict(data)

    return labels