import base64
import io
import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from fuzzywuzzy import fuzz  # Import fuzzywuzzy's fuzz module

# Global variables to store FAISS index and text chunks
faiss_index = None
chunks = None
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load model once

def extract_text_from_base64_pdf(pdf_dict):
    """
    Extracts text from a Base64-encoded PDF.
    """
    try:
        # Extract the Base64 string from the dictionary
        base64_pdf = pdf_dict.get('document', '')
        if not base64_pdf:
            return "Error: No PDF data found in the input."

        # Decode the Base64 PDF
        if "," in base64_pdf:
            base64_pdf = base64_pdf.split(",")[1]  # Remove the "data:application/pdf;base64," prefix
        pdf_bytes = base64.b64decode(base64_pdf)

        # Use BytesIO to simulate a file-like object
        pdf_stream = io.BytesIO(pdf_bytes)

        # Open the PDF with PyMuPDF
        doc = fitz.open(stream=pdf_stream, filetype="pdf")

        # Extract text from each page
        extracted_text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            extracted_text += page.get_text()

        # Close the document to free resources
        doc.close()

        if not extracted_text.strip():
            return "Error: PDF contains no extractable text."

        return extracted_text

    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def preprocess_text(text):
    """
    Splits text into manageable chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Customize as needed
        chunk_overlap=20,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def embed_text(chunks):
    """
    Convert text chunks into embeddings using SentenceTransformer.
    """
    embeddings = model.encode(chunks)
    return embeddings


def create_faiss_index(embeddings):
    """
    Create a FAISS index from the embeddings and return the index.
    """
    dimension = embeddings.shape[1]  # This depends on the embedding dimension
    faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    faiss_index.add(embeddings.astype(np.float32))  # Add embeddings to the FAISS index
    return faiss_index


def retrieve_similar_chunks(query, faiss_index, chunks):
    """
    Retrieve the most similar chunks to the query from the FAISS index.
    """
    # Convert the query into an embedding
    query_embedding = model.encode([query])

    # Perform the similarity search
    D, I = faiss_index.search(np.array(query_embedding).astype(np.float32), k=2)  # Adjust k for top 2

    # Retrieve the corresponding chunks
    similar_chunks = [chunks[i] for i in I[0]]
    return similar_chunks


def fuzzy_match_with_chunks(query, chunks):
    """
    Perform fuzzy matching to score and rank the most relevant chunks.
    """
    scores = []
    for chunk in chunks:
        score = fuzz.partial_ratio(query, chunk)  # You can adjust this method depending on your needs
        scores.append((chunk, score))

    # Sort chunks based on fuzzy match score
    sorted_chunks = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_chunks


def hybrid_search(query, faiss_index, chunks):
    """
    Combine FAISS retrieval with fuzzy matching to return 2 FAISS-based results and 2 fuzzy-based results.
    """
    # Step 1: Retrieve top-2 similar chunks using FAISS
    faiss_similar_chunks = retrieve_similar_chunks(query, faiss_index, chunks)

    # Step 2: Perform fuzzy matching on the FAISS retrieved chunks
    fuzzy_matched_chunks = fuzzy_match_with_chunks(query, faiss_similar_chunks)

    # Step 3: Select top 2 results from FAISS and top 2 results from fuzzy matching
    faiss_results = faiss_similar_chunks[:2]  # Top 2 FAISS results
    fuzzy_results = [chunk[0] for chunk in fuzzy_matched_chunks[:2]]  # Top 2 fuzzy matching results

    # Combine FAISS and fuzzy matching results
    combined_results = faiss_results + fuzzy_results

    return combined_results


def upload_pdf(pdf_dict):
    """
    Handle PDF upload, extract text, and create a new FAISS index.
    """
    global faiss_index, chunks
    file_content = pdf_dict.get("document", "")
    if file_content:

        # Step 1: Extract text from the PDF
        extracted_text = extract_text_from_base64_pdf(pdf_dict)
        print("extracted_text",extracted_text)

        # Step 2: Preprocess the extracted text into chunks
        chunks = preprocess_text(extracted_text)

        # Step 3: Create embeddings and FAISS index
        embeddings = embed_text(chunks)
        faiss_index = create_faiss_index(embeddings)
        return "PDF uploaded and FAISS index created successfully."

    else:
        return "not PDF  and FAISS index in the db."


def main(query):
    """
    This function will be called every time a query is made.
    """
    # If FAISS index and chunks exist, continue with the search
    if faiss_index is None or chunks is None:
        print("no index and chunk")
        return "Error: Please upload a PDF first."

    # Retrieve the most similar chunks using hybrid search
    print("query",query)
    similar_chunks = hybrid_search(query, faiss_index, chunks)

    return similar_chunks


# Example usage:
# # 1. Upload PDF first
# pdf_dict = {'document': 'data:application/pdf;base64,JVBERi0xLjcKJcK1wrYKCjEgMCBvYmoKPDwvVHlwZS9DYXRhbG9nL1BhZ2VzIDIgMCBSPj4KZW5kb2JqCgoyIDAgb2JqCjw8L1R5cGUvUGFnZXMvQ291bnQgMS9LaWRzWzQgMCBSXT4+CmVuZG9iagoKMyAwIG9iago8PC9Gb250PDwvaGVsdiA1IDAgUj4+Pj4KZW5kb2JqCgo0IDAgb2JqCjw8L1R5cGUvUGFnZS9NZWRpYUJveFswIDAgNTk1IDg0Ml0vUm90YXRlIDAvUmVzb3VyY2VzIDMgMCBSL1BhcmVudCAyIDAgUi9Db250ZW50c1s2IDAgUl0+PgplbmRvYmoKCjUgMCBvYmoKPDwvVHlwZS9Gb250L1N1YnR5cGUvVHlwZTEvQmFzZUZvbnQvSGVsdmV0aWNhL0VuY29kaW5nL1dpbkFuc2lFbmNvZGluZz4+CmVuZG9iagoKNiAwIG9iago8PC9MZW5ndGggNzY5L0ZpbHRlci9GbGF0ZURlY29kZT4+CnN0cmVhbQp42o1VO8/UMBDs8ytSIwG2d71OJESBQEh0oHSI4u6SQAEFFPx+ZsfJXXK5D1CUp/cxOzveND+bN0MT24Ajtjm0pU/t8KN5+W36/ruNoR1mroX20/vl4dfX9vOrrDYXtWiXFPzZzpZtKiKnFCRLJ93rL8OHJrTPo7woGkKI7fC2uXOUEuxifdHFMdJpeIYEtyd4wAJ22W1uPnt7fM824ogllV5OO//Ocgrwi6UvQu+QAq6CqwFPSkFP8J/cTpL4qb5WEkobsdrB1yP0lot4bh3xPJl6DGTIJsgywfqCs1Q7WCEXMDn+bJpQsyVEEr55tAk2yamDDzgpGeiz2+nZK8H3SLbwliZmYi3wGMnDXJCJX1hNDh4B9ifm7OxMVqJXmAXfE6M6JmTDEWHX4zoRW0RerRlpL14H2uodYNWl1LiwOyDQXp2XTkfwh97gVFyJWyvqE7hx1mr1M6ufDZHxDBYcJTx6x7J0IVVczjTu3VqlRoUvkHl1i2ddU3EcGTbw71BCp7JB4H2UWim4QneLAZHWKmFlVIEyp9c/l2XliiHWHuLtfLTa8NMTCc40PVK1Pq27y1/7WIi/ELvwOS9KcWa5ChTujZjnAg7AJ1hduLwhLJudSAYUObwXvXeGkWczxlD3Fb8LYyTuggy9as26apnsoBpywf22iSKSFyR7LT/O7XHFd4SvX9+xG7mXXaXdrSuFvcRedlxSo9TKFsuqACqEenUmR2cgjVYV/3QOejmTQA/s6NAmGrRotQZy2QPxI/TCupR7Dzt8V+l1ylhiR2pvHuxzqnzV4f3+22A6KuI45Q4I43U29eYsCScEp+mmh4ksWO2lZ8xJ42bdsJpdydSJz+rMKVJ7cVSE20NVVVe7PYZ9WbplUiRylXeaNHbD7xOn1n9o6J8quaw2+6lg61S4PKGRO+x33EUgM66L9+mJf9zaO5+F9V+2Zq/fmbMEaseVMOKPwV76nFj+LzcUHqXyNbmmlj0bEbevkwE+uaqsznfEjyt3B4zvhuZj8wd6WtizCmVuZHN0cmVhbQplbmRvYmoKCnhyZWYKMCA3CjAwMDAwMDAwMDAgMDAwMDEgZiAKMDAwMDAwMDAxNiAwMDAwMCBuIAowMDAwMDAwMDYyIDAwMDAwIG4gCjAwMDAwMDAxMTQgMDAwMDAgbiAKMDAwMDAwMDE1NSAwMDAwMCBuIAowMDAwMDAwMjYyIDAwMDAwIG4gCjAwMDAwMDAzNTEgMDAwMDAgbiAKCnRyYWlsZXIKPDwvU2l6ZSA3L1Jvb3QgMSAwIFIvSURbPDNCNDA1NEMzOUNDM0I2QzNBQUMzOEIzMkMyQjhDMkEzPjxCMUUwNzI0MEQ3OTFFQkUxOTg0REI0RjAxODcxRkM3Mj5dPj4Kc3RhcnR4cmVmCjExODkKJSVFT0YK'}
# upload_pdf(pdf_dict)

# 2. Query for information
# query = "What is purajith"
# results = main(query)
# print(results)
