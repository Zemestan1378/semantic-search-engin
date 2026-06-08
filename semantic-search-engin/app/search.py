from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

from document import load_documents

documents = load_documents(
    "data/knowledge.txt"
)
doc_embeddings = model.encode(documents)

query = input("Search: ")

query_embedding = model.encode([query])

scores = cosine_similarity(
    query_embedding,
    doc_embeddings
)

best_index = scores.argmax()

print("\nBest Match:")
print(documents[best_index])

print("\nSimilarity:")
print(scores[0][best_index])

