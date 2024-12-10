import os
import pdfplumber
from flask import Flask, request, render_template, redirect, flash, session, jsonify
import secrets
import uuid
from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret
import time

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Set a secret key for session management

# Initialize variables
UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the upload folder if it doesn't exist

# Update prompt template for the LLM
prompt_template = """
Anda adalah asisten yang membantu.
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

Pertanyaan: {{ question }}

Berikan analisis singkat dan relevan berdasarkan informasi dalam dokumen dengan langsung menjawab pertanyaan singkat, padat dan jelas dengan bahasa Indonesia. 
"""

# Limit the number of characters or tokens from the extracted text
MAX_TEXT_LENGTH = 5000  # Set a maximum length for the text
MAX_RETRIES = 3  # Set maximum retries for API calls

@app.route("/", methods=["GET"])
def index():
    # Redirect to the upload page
    return redirect("/upload")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    session.pop("messages", None)  # Clear previous messages from session

    if request.method == "POST":
        # Reset variables here
        global document_store, retriever, prompt_builder, llm, rag_pipeline
        document_store = InMemoryDocumentStore()  # Create a new document store
        retriever = InMemoryBM25Retriever(document_store=document_store)  # Create a new retriever
        prompt_builder = PromptBuilder(template=prompt_template)  # Create a new prompt builder
        llm = OpenAIGenerator(
            api_key=Secret.from_token(os.getenv("GROQ_API_KEY")),  # Get API key from environment variable
            api_base_url="https://api.groq.com/openai/v1",  # Base URL for the API
            model="llama-3.3-70b-versatile",  # Specify the model to use
            generation_kwargs={"max_tokens": 1024}  # Set generation parameters
        )
        rag_pipeline = Pipeline()  # Create a new pipeline
        rag_pipeline.add_component("retriever", retriever)  # Add the retriever component
        rag_pipeline.add_component("prompt_builder", prompt_builder)  # Add the prompt builder component
        rag_pipeline.add_component("llm", llm)  # Add the LLM component
        rag_pipeline.connect("retriever", "prompt_builder.documents")  # Connect retriever to prompt builder
        rag_pipeline.connect("prompt_builder", "llm")  # Connect prompt builder to LLM

        uploaded_file = request.files.get("file")  # Get the uploaded file
        if uploaded_file and uploaded_file.filename.endswith(".pdf"):  # Check if the file is a PDF
            # Remove existing files in the upload folder
            for existing_file in os.listdir(UPLOAD_FOLDER):
                os.remove(os.path.join(UPLOAD_FOLDER, existing_file))

            filename = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)  # Create the full file path
            uploaded_file.save(filename)  # Save the uploaded file

            # Extract text from the PDF
            text = ""
            with pdfplumber.open(filename) as pdf:  # Open the PDF file
                for page in pdf.pages:  # Iterate through each page
                    page_text = page.extract_text()  # Extract text from the page
                    if page_text:  # Check if text was extracted
                        text += page_text + "\n\n"  # Append extracted text
                    else:
                        print(f"Tidak ada teks yang ditemukan di halaman {pdf.pages.index(page) + 1}.")  # Log if no text found

            # Log the extracted text length
            print(f"Panjang teks yang diekstrak: {len(text)} karakter.")

            # Truncate the text to the maximum length
            text = text[:MAX_TEXT_LENGTH] if len(text) > MAX_TEXT_LENGTH else text

            # Write the document to the document store
            document_store.write_documents([Document(content=text, meta={"source": filename, "id": str(uuid.uuid4())})])
            flash('File berhasil diunggah dan diproses.')  # Flash a success message

            return redirect("/chat")  # Redirect to the chat page

    return render_template("upload.html")  # Render the upload page

@app.route("/chat", methods=["GET"])
def chat():
    if "messages" not in session:
        session["messages"] = []  # Initialize messages in session if not present

    messages = session["messages"]  # Get messages from session
    session.pop('_flashes', None)  # Clear flash messages

    return render_template("chat.html", messages=messages)  # Render the chat page with messages

@app.route("/send_message", methods=["POST"])
def send_message():
    if "messages" not in session:
        session["messages"] = []  # Initialize messages in session if not present

    messages = session["messages"]  # Get messages from session
    question = request.json.get("question")  # Get the user's question from the request
    answer = "Maaf, saya tidak memahami pertanyaan Anda."  # Default answer

    if question:
        for attempt in range(MAX_RETRIES):  # Retry logic for API calls
            try:
                results = rag_pipeline.run(
                    {
                        "retriever": {"query": question},  # Query the retriever
                        "prompt_builder": {"question": question},  # Pass the question to the prompt builder
                    }
                )
                # Check if the document has relevant content
                if results["llm"]["replies"] and results["llm"]["replies"][0]:
                    answer = results["llm"]["replies"][0]  # Get the answer from LLM
                else:
                    answer = "Dokumen tidak mengandung informasi yang dapat menjawab pertanyaan ini."  # Custom message for no content
                break  # Exit loop if successful
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    answer = "Maaf, layanan tidak tersedia saat ini. Silakan coba lagi nanti."  # Error message
                    print(f"Error during API call: {e}")  # Log the error

        # Append user question and bot answer to messages
        messages.append({"type": "user", "content": question})
        messages.append({"type": "bot", "content": answer})
        session["messages"] = messages  # Save messages back to session

        return jsonify({"answer": answer})  # Return the answer as JSON

    return jsonify({"answer": answer})  # Return default answer if no question

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode
