import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings

# Suppress FutureWarnings related to tokenization
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Extract and Prepare the Dataset
url = "https://raw.githubusercontent.com/datum-oracle/netflix-movie-titles/main/titles.csv"
movies_df = pd.read_csv(url)

# Combine relevant columns into a single 'combined_text' column
movies_df['combined_text'] = movies_df[['title', 'description', 'genres']].fillna('').agg(' '.join, axis=1)

# Ensure that the combined text column has no NaN values
movies_df['combined_text'] = movies_df['combined_text'].fillna('')

# Add mock 'score' and 'votes' columns for sorting demonstration purposes
np.random.seed(42)  # For reproducibility
movies_df['score'] = np.random.uniform(1, 10, len(movies_df))
movies_df['votes'] = np.random.randint(100, 10000, len(movies_df))

# Step 2: Transform
# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the combined text column
embeddings = model.encode(movies_df['combined_text'].tolist())

# Convert embeddings to numpy array
embeddings_np = np.array(embeddings)

# Step 3: Load into FAISS
# Initialize the FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Add embeddings to the FAISS index
index.add(embeddings_np)

print("Embeddings indexed in FAISS.")

# Function to find movies based on search term
def find_movies(search_term, sort_by=None):
    # Transform the search term into an embedding
    search_embedding = model.encode([search_term])[0]

    # Convert the search embedding to a numpy array
    search_embedding_np = np.array([search_embedding])

    # Perform the search in the FAISS index (retrieving top 5 results)
    D, I = index.search(search_embedding_np, k=5)  # k is the number of nearest neighbors

    # Get the indices of the top 5 closest embeddings
    closest_indices = I[0]

    # Retrieve the matching movies based on the indices
    search_filtered = movies_df.iloc[closest_indices][['title', 'genres', 'score', 'votes']]

    # Optional sorting based on user input
    if sort_by and sort_by in ['score', 'votes']:
        search_filtered = search_filtered.sort_values(by=sort_by, ascending=False)

    return search_filtered

# Loop to keep prompting the user for input
while True:
    search_term = input("Enter a search term (or enter '0' to exit): ")

    if search_term == '0':
        print("Exiting the program.")
        break

    # Optional: Ask the user if they want to sort the results
    sort_by = input("Sort by 'score', 'votes', or press Enter to skip sorting: ").strip().lower()
    if sort_by not in ['score', 'votes', '']:
        print("Invalid sort option. Results will not be sorted.")
        sort_by = None  # No sorting if invalid input

    # Get the movies matching the search term
    matching_movies = find_movies(search_term, sort_by=sort_by)

    # Display the matching movie titles
    if not matching_movies.empty:
        print(f"\nMovies matching '{search_term}':")
        print(matching_movies)
    else:
        print(f"No movies found for '{search_term}'.")
