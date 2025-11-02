from flask import Flask, request, jsonify, render_template
from rag import extract_text_from_base64_pdf, main, upload_pdf
from flask import Flask, request, jsonify, render_template, send_file, session
import fitz
from gpt_bot import bot ,memory_clean
from lama_bot import chat_with_llama,rag_summary
app = Flask(__name__)
from PyPDF2 import PdfReader
from summarizer import process_pdf
import os
app.secret_key = "123"  # Needed for session encryption


llm = ["llama3.3-70b", "llama3.1-8b","llama3.2-3b","gemma2-9b"]
gpt_llm = ["gpt-3.5-turbo","gpt-4o","gpt-4o-mini"]
app.config['UPLOAD_FOLDER'] = 'uploads'
# Route for the main page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_model', methods=['POST'])
def set_model():
    data = request.get_json()  # Parse incoming JSON data
    model_selection = data.get('model', 'gpt-4o-mini')  # Default to 'gpt-4o-mini' if no model is provided
    session['selected_model'] = model_selection  # Store the model in the session
    print("Model stored in session:", model_selection)
    return jsonify({"success": True, "selected_model": model_selection})

@app.route('/get_model', methods=['GET'])
def get_model():
    # Retrieve the model from the session
    model_selection = session.get('selected_model', 'gpt-4o-mini')  # Default to 'gpt-4o-mini'
    return jsonify({"selected_model": model_selection})

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    # Default memory configurations
    memory_limit = 4
    memory_flag = "yes"
    user_input = "who are you"
    clear_memory = "no"

    if request.method == 'POST':
        data = request.get_json()  # Parse incoming JSON data
        model_selection = data['model']  # Get the selected model

        
        if model_selection in gpt_llm: 
            print("gpt is selected",model_selection)


            # Check for specific keys in the POST data
            if "clear_memory" in data:
                clear_memory = data["clear_memory"]
                # print("clear_memory",clear_memory)
                if clear_memory == "yes":
                    # Logic to clear memory (e.g., reset memory state in your bot function)
                    print("Memory cleared.")
                    memory_clean(clear_memory)
                    return jsonify({"response": "Memory has been cleared."})

            memory_limit = int(data.get("memory_limit", memory_limit))
            memory_flag = data.get("memory_flag", memory_flag)
            user_message = data.get("message", "")  # Get user's chat message

            if not user_message:
                return jsonify({"response": "Please provide a message."})

            # Pass memory settings to the bot function

            bot_response = bot(memory_limit, memory_flag, user_message,clear_memory)
            return jsonify({"response": bot_response})
        elif model_selection in llm:
            print("llama is selected",model_selection)
            user_message = data.get("message", "")  # Get user's chat message
            bot_response= chat_with_llama(user_message, model_selection)
            return jsonify({"response": bot_response})



    return render_template('bot.html')


# Route for the RAG page
@app.route('/rag', methods=['GET', 'POST'])
def rag():
    if request.method == 'POST':
        data = request.get_json()  # Parse the incoming JSON data
        # model_selection = data['model']  # Get the selected model

        model_selection = data.get("model", "")  # Get the user's message



        if data is not None:  # Check if data is not None    
            extract =upload_pdf(data)

        print(str(extract))
        user_message = data.get("message", "")  # Get the user's message
        result =main(user_message)
        prompt = (
            f"You are an AI assistant specialized in legal research, assisting with a Retrieval-Augmented Generation (RAG) system. "
            f"Based on the provided context, answer the user's query in a clear and accurate manner.\n\n"
            f"Context:\n\"{result}\"\n\n"
            f"User Query:\n\"{user_message}\"\n\n"
            f"Provide a concise and detailed response:"
        )
        print("user message",user_message)
        print("      result",result)

        
        if not user_message:
            return jsonify({"response": "Please provide a message."})
        memory_flag ="no"
        memory_limit ="1"
        clear_memory ="no"
        if model_selection in gpt_llm: 
            print("gpt is selected",model_selection)
            bot_response = bot(memory_limit, memory_flag, prompt,clear_memory)
            print("bot_response",bot_response)

        elif model_selection in llm:
            print("llama is selected",model_selection)
            bot_response= chat_with_llama(prompt, model_selection)
            print("bot_response",bot_response)
            return jsonify({"response": bot_response})

        # Example logic for RAG response
        rag_response = f"RAG response for your query: {bot_response}"
        return jsonify({"response": bot_response})


    return render_template('rag.html')
# app.secret_key = '123456'  # Needed for Flask's session functionality
content = ""

@app.route('/scrap', methods=['GET', 'POST'])
def scrap():
    global content  # Use the global content variable

    if request.method == 'GET':
        return render_template('summary.html')

    action = request.headers.get('action')
    # model     = data.get('model')
    # print("model",model)
    model_selection = session.get('selected_model', 'gpt-4o-mini')

    if model_selection in gpt_llm: 
        print("gpt is selected",model_selection)

    print("get_model():",model_selection)

    if action == 'upload':
        file = request.files.get('pdf')
        if not file or not file.filename.endswith('.pdf'):
            return jsonify({"response": "Invalid file. Please upload a PDF."}), 400

        # Read the PDF content using PyMuPDF
        try:
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            for page in pdf_document:
                content += page.get_text()
            default_prompt =" Summarize the following document into a concise and meaningful overview, ensuring all key points and essential details are included:"
            if model_selection in gpt_llm: 
                print("gpt is selected",model_selection)
                summaries, total_tokens = process_pdf(
                content,default_prompt,model=model_selection  # Adjust as needed
            )
            elif model_selection in llm:
                query = str(default_prompt)+str(content) 
                summaries = rag_summary(query, model_selection)


            # Return the PDF content in the response
            return jsonify({"response": "PDF uploaded successfully.", "pdf_content": summaries})
        except Exception as e:
            return jsonify({"response": f"Error processing PDF: {str(e)}"}), 500

    elif action == 'summarize':
        data = request.json
        custom_prompt = data.get('prompt', 'Default summary prompt')
        print(custom_prompt)
        pdf_content = data.get('pdf_content')

        if not pdf_content:
            return jsonify({"response": "No PDF content provided."}), 400
        # custom_prompt ="Summarize based on process"
    #     summaries, total_tokens = process_pdf(pdf_content,
    #     custom_prompt,model=model_selection  # Adjust as needed
    # )
        if model_selection in gpt_llm: 
            print("gpt is selected",model_selection)
            summaries, total_tokens = process_pdf(
            content,custom_prompt,model=model_selection  # Adjust as needed
        )
        elif model_selection in llm:
            query = custom_prompt+str(content)
            summaries = rag_summary(query, model_selection)

        # Generate a simple summary (replace this with your summarization logic)
        summary = f"Summary based on prompt: {custom_prompt}\n\n{summaries}..."  # First 500 characters
        return jsonify({"summary": summary})

    elif action == 'download':
        data = request.json
        summary = data.get('summary')

        if not summary:
            return jsonify({"response": "No summary provided."}), 400

        # Create a downloadable text file
        summary_file = BytesIO()
        summary_file.write(summary.encode('utf-8'))
        summary_file.seek(0)
        return send_file(summary_file, as_attachment=True, download_name="summary.txt")

    elif action == 'reset':
        return jsonify({"response": "Reset action is unnecessary without session usage."})

    return jsonify({"response": "Invalid request."}), 400


# # Ensure upload folder exists
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# # Utility functions
# def pdf_to_text(pdf_path, output_path):
#     """Convert PDF to Text and save as PDF."""
#     reader = PdfReader(pdf_path)
#     extracted_text = ""
#     for page in reader.pages:
#         extracted_text += page.extract_text() + "\n"
#     return save_text_as_pdf(extracted_text, output_path)

# def pdf_to_word(pdf_path, output_path):
#     """Convert PDF to Word."""
#     reader = PdfReader(pdf_path)
#     doc = Document()
#     for page in reader.pages:
#         doc.add_paragraph(page.extract_text())
#     doc.save(output_path)
#     return output_path

# def img_to_text(img_path, output_path):
#     """Convert Image to Text and save as PDF."""
#     image = Image.open(img_path)
#     extracted_text = pytesseract.image_to_string(image)
#     return save_text_as_pdf(extracted_text, output_path)

# def save_text_as_pdf(text, output_path):
#     """Save extracted text as a PDF file."""
#     c = canvas.Canvas(output_path)
#     c.drawString(100, 750, "Extracted Text:")
#     text_lines = text.splitlines()
#     y = 730  # Initial vertical position
#     for line in text_lines:
#         if y < 50:  # Avoid writing beyond the bottom margin
#             c.showPage()
#             y = 750
#         c.drawString(50, y, line)
#         y -= 15
#     c.save()
#     return output_path

# @app.route('/convert', methods=['GET', 'POST'])
# def convert():
#     """File conversion logic."""
#     if request.method == 'GET':
#         return render_template('conversion.html')

#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     conversion_type = request.form.get('conversionType', '')

#     if not file or not conversion_type:
#         return jsonify({'error': 'Invalid input'}), 400

#     # Generate unique paths
#     file_extension = os.path.splitext(file.filename)[1]
#     file_id = str(uuid.uuid4())
#     input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}{file_extension}")
#     file.save(input_path)

#     try:
#         if conversion_type == 'pdf_to_text':
#             output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_output.pdf")
#             output_file = pdf_to_text(input_path, output_path)
#         elif conversion_type == 'pdf_to_word':
#             output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_output.docx")
#             output_file = pdf_to_word(input_path, output_path)
#         elif conversion_type == 'img_to_text':
#             output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_output.pdf")
#             output_file = img_to_text(input_path, output_path)
#         else:
#             return jsonify({'error': 'Invalid conversion type'}), 400

#         return send_file(output_file, as_attachment=True)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
