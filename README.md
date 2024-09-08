# Movie Search Engine with FAISS and Sentence Transformers

## Overview

This project implements a simple movie search engine using natural language processing and FAISS (Facebook AI Similarity Search). The search engine allows users to find movies based on a search term and optionally sort the results by score or votes. The engine is built using Python, Pandas, the SentenceTransformers library, and FAISS for efficient similarity search.

## Requirements

The requirements.txt file includes:

pandas
sentence-transformers
faiss-cpu
numpy



## Usage

### Run the Application:

```bash
python movie.py
This will start the application, and you will be prompted to enter a search term.

Input Search Term:
Type any word or phrase to search for movies that match. You can also enter 0 to exit the application.

Sort the Results (Optional):
After entering the search term, you can choose to sort the results by score or votes. If you don't want to sort the results, just press Enter.

View the Results:
The application will display the top 5 movies that match your search term, along with their genres, scores, and votes.

Project Structure
Copy
├── movie.py        # Main Python script
├── README.md              # Documentation
├── requirements.txt       # Python dependencies
└── data
    └── titles.csv         # Movie dataset (downloaded automatically from URL)
Code Explanation
1. Data Extraction and Preparation
Dataset: The movie data is fetched from a CSV file available online.
Data Cleaning: Columns like title, description, and genres are combined into a single combined_text column to form the input text for the search engine.
Mock Data: The script adds mock score and votes columns to simulate sorting functionality.

2. Text Embedding using Sentence Transformers
Model Initialization: The SentenceTransformer model all-MiniLM-L6-v2 is used to convert text data into embeddings.
Embedding Generation: The combined_text column is transformed into numerical embeddings that capture semantic meaning.

3. FAISS for Similarity Search
Index Initialization: A FAISS index is created to store the embeddings.
Indexing: The embeddings are added to the FAISS index for efficient similarity searches.

4. Search Functionality
Search Term Embedding: When a user inputs a search term, it is also converted into an embedding.
Search in FAISS: The FAISS index is queried to find the top 5 closest embeddings to the search term.
Result Display: The matching movies are displayed and can be optionally sorted by score or votes.

5. User Interaction
Loop for Continuous Input: The script continuously prompts the user for search terms until the user decides to exit by entering 0.
