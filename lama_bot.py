import json
from llamaapi import LlamaAPI
from transformers import LlamaTokenizer
import tiktoken

# Initialize the SDK
llama = LlamaAPI("LA-37f51aa1483e4d5a9d35704906fd7b23ce73f23c9c004760a5b80026fc609282")

def query_llama_api(user_message,model):
    print("model",model)
    api_request_json = {
        "model": model,  # You can change this to a LLaMA model if necessary
        "messages": [{"role": "user", "content": user_message}],
        "stream": False
    }
    
    response = llama.run(api_request_json)

    # Parse the response content
    try:
        response_data = response.json()  # Extract JSON content
    except Exception as e:
        response_data = None
        print(f"Failed to parse response JSON: {e}")

    print(response_data)  # Debug to see the actual structure
    return response.json()

def chat_with_llama(query, model):
    # Get user input
    user_input = query


    # Query LLaMA API with the user input only (no history)
    response = query_llama_api(user_input,model)

    # Extract response content
    if "choices" in response and len(response["choices"]) > 0:
        bot_response = response["choices"][0].get("message", {}).get("content", "No response received.")
    else:
        bot_response = "Unexpected response format from API."
    return bot_response

    # Display the conversation
    print(f"Bot: {bot_response}")

# Run the chat function

from transformers import LlamaTokenizer
import tiktoken

def split_into_chunks(text, model, max_tokens):
    """
    Splits the input text into chunks that fit within the token limit using OpenAI's `tiktoken`.
    """
    try:
        # Get the encoding for the model
        encoding = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        # Fallback to a default encoding if the model is not recognized
        print(f"Model '{model}' not recognized. Falling back to 'cl100k_base'.")
        encoding = tiktoken.get_encoding("cl100k_base")

    # Tokenize the input text
    tokens = encoding.encode(text)
    
    # Split tokens into chunks
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    # Decode tokens back to text chunks
    text_chunks = [encoding.decode(chunk) for chunk in chunks]
    
    return text_chunks


def process_chunk(chunk, model):
    """
    Process a single chunk of text using the LLaMA API.
    """
    print(f"Processing chunk with {len(chunk.split())} tokens...")  # Estimate tokens in the chunk
    api_request_json = {
        "model": model,
        "messages": [{"role": "user", "content": chunk}],
        "stream": False
    }

    # Simulated llama.run for testing
    try:
        response = llama.run(api_request_json)
        response_data = response.json()
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return "Error in processing chunk."

def rag_summary(user_message, model):
    """
    Process a long input message for RAG-based summarization.
    """
    # Define token limits for each model
    model_token_limits = {
        "llama3.3-70b": 6000,
        "llama3.1-8b": 3000,
        "llama3.2-3b": 1800,
        "gemma2-9b": 6000
    }

    # Get the token limit for the selected model
    max_tokens = model_token_limits.get(model, 4000) - 200  # Reserve space for the response

    # Split the input into manageable chunks
    chunks = split_into_chunks(user_message, model, max_tokens)

    # Process each chunk and collect summaries
    summaries = [process_chunk(chunk, model) for chunk in chunks]

    # Combine all summaries into one final summary
    final_summary = " ".join(summaries)
    return final_summary



# Example usage
query = "summarize this prompt' im purajith kumar, i m a data scientist'"  # Replace with actual text
model = "llama3.1-8b"  # Replace with the desired model

# # Generate a RAG-based summary
# summary = rag_summary(query, model)
# print("Final Summary:")
# print(summary)
