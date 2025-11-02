import openai
import tiktoken
from PyPDF2 import PdfReader

# Set up OpenAI API key
openai.api_key = "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"

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def tokenize_text(text, model="gpt-4-turbo"):
    """Tokenize text using OpenAI's tokenizer."""
    model="gpt-4-turbo"
    encoding = tiktoken.encoding_for_model(model)
    print(len(encoding.encode(text)))

    return encoding.encode(text)

def decode_tokens(tokens, model="gpt-4-turbo"):
    """Decode tokens back to text."""
    model="gpt-4-turbo"
    encoding = tiktoken.encoding_for_model(model)
    return encoding.decode(tokens)

def summarize_text(chunk, prompt, model="gpt-4o-mini"):
    """Summarize a text chunk with a given prompt."""
    print("model")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\n{chunk}"}
        ]
    )
    return response['choices'][0]['message']['content']

def process_pdf(text, prompt, model="gpt-4o"):
    """Process the PDF for summarization."""
    tokens = tokenize_text(text, model=model)
    print(model)
    # Chunking logic
    chunk_size = 1000
    token_chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    summaries = []
    total_tokens = 0
    
    for i, chunk_tokens in enumerate(token_chunks):
        # Decode tokens back to text
        chunk_text = decode_tokens(chunk_tokens, model=model)
        summary_prompt = prompt
        
        summary = summarize_text(chunk_text, summary_prompt, model=model)
        summaries.append((i + 1, summary))  # (Summary Number, Summary Text)
        total_tokens += len(chunk_tokens)
    
    return summaries, total_tokens

# def main():
#     # User inputs
#     pdf_path = "21page_pdf.pdf"
#     default_prompt = "Please summarize the following text:"
#     custom_prompt = "no"
#     use_custom_prompt = bool(custom_prompt)  # Use custom prompt if provided

#     # Process the PDF
#     try:
#         summaries, total_tokens = process_pdf(
#             text,prompt,model)

#         # Display results
#         print("\nSummarization Results:")
#         for summary_number, summary in summaries:
#             print(f"\nSummary #{summary_number}:\n{summary}\n")

#         print(f"\nTotal tokens processed: {total_tokens}")
#         print(f"Number of summaries: {len(summaries)}")

#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

# # kirpal-singh-v-govt-of-india-575735.pdf 