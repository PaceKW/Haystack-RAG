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
import markdown  # Import library markdown

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Initialize variables
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize document store and components
document_store = InMemoryDocumentStore()
retriever = InMemoryBM25Retriever(document_store=document_store)

# Update prompt template to include specific analysis instructions and focus
prompt_template = """
Diberikan dokumen-dokumen berikut, lupakan dokumen sebelumnya dan fokuslah pada dokumen ini. 
Anda adalah seorang pembaca yang kritis, memiliki kemampuan berpikir analitis, dan dapat menyimpulkan informasi dengan baik.

Dokumen:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

Pertanyaan: {{ question }}

"""
prompt_builder = PromptBuilder(template=prompt_template)

# Use Secret to wrap the API key
llm = OpenAIGenerator(
    api_key=Secret.from_token(os.getenv("GROQ_API_KEY")),
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    generation_kwargs={"max_tokens": 512}
)

# Pipeline RAG
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Limit the number of characters or tokens from the extracted text
MAX_TEXT_LENGTH = 5000  # Set a maximum length for the text

@app.route("/", methods=["GET"])
def index():
    return redirect("/upload")  # Redirect to the upload page

@app.route("/upload", methods=["GET", "POST"])
def upload():
    # Clear the session messages for a new chat
    session.pop("messages", None)  # Clear previous messages

    if request.method == "POST":
        # Handle file upload
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename.endswith(".pdf"):
            # Hapus file dokumen sebelumnya jika ada
            existing_files = os.listdir(UPLOAD_FOLDER)
            for existing_file in existing_files:
                os.remove(os.path.join(UPLOAD_FOLDER, existing_file))  # Hapus file yang ada

            filename = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(filename)

            # Extract text from the PDF
            text = ""
            with pdfplumber.open(filename) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"

            # Truncate the text to the maximum length
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]

            # Write the document to the document store
            document_store.write_documents([Document(content=text, meta={"source": filename, "id": str(uuid.uuid4())})])
            flash('File berhasil diunggah dan diproses.')  # This will be shown only once

            return redirect("/chat")  # Redirect to chat page

    return render_template("upload.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    # Ambil daftar pesan sebelumnya dari session
    if "messages" not in session:
        session["messages"] = []

    messages = session["messages"]

    # Clear flash messages to avoid showing them on new chat
    session.pop('_flashes', None)

    if request.method == "POST":
        question = request.form.get("question")  # Ambil pertanyaan pengguna
        if question:
            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Jalankan pipeline RAG
                    results = rag_pipeline.run(
                        {
                            "retriever": {"query": question},
                            "prompt_builder": {"question": question},
                        }
                    )

                    # Ambil jawaban dari LLM
                    answer = results["llm"]["replies"][0] if "replies" in results["llm"] else "Maaf, saya tidak memahami pertanyaan Anda."
                    break  # Exit loop if successful
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        answer = "Maaf, layanan tidak tersedia saat ini. Silakan coba lagi nanti."
                        # Log the error (optional)
                        print(f"Error during API call: {e}")

            # Tambahkan pesan baru ke dalam riwayat
            messages.append({"type": "user", "content": question})
            messages.append({"type": "bot", "content": answer})

            # Simpan kembali riwayat ke dalam session
            session["messages"] = messages

    return render_template("chat.html", messages=messages)

@app.route("/send_message", methods=["POST"])
def send_message():
    question = request.json.get("question")  # Get the user's question from the AJAX request
    answer = "Maaf, saya tidak memahami pertanyaan Anda."  # Default answer

    if question:
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Run the RAG pipeline
                results = rag_pipeline.run(
                    {
                        "retriever": {"query": question},
                        "prompt_builder": {"question": question},
                    }
                )

                # Get the answer from the LLM
                answer = results["llm"]["replies"][0] if "replies" in results["llm"] else answer
                break  # Exit loop if successful
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    answer = "Maaf, layanan tidak tersedia saat ini. Silakan coba lagi nanti."
                    print(f"Error during API call: {e}")

    return jsonify({"answer": answer})  # Return the answer as JSON

if __name__ == "__main__":
    app.run(debug=True)
