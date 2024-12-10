# Haystack-RAG: PDF Chatbot Application

Welcome to **Haystack-RAG**, a powerful web application that allows users to upload PDF documents and interact with a chatbot capable of answering questions based on the content of those documents. Built with Flask and integrated with OpenAI's language model, this application provides a seamless experience for extracting information from PDFs.

## Table of Contents

- [Features](#features)
- [How to Use](#how-to-use)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features

- **PDF Upload**: Easily upload PDF files and extract text for interaction.
- **Interactive Chat Interface**: Ask questions about the uploaded documents in a user-friendly chat format.
- **Intelligent Responses**: Utilizes a retriever and OpenAI's language model to provide accurate and relevant answers.
- **Drag-and-Drop Functionality**: Simplified file upload process for a better user experience.


## How to Use

1. **Clone this repository:**
   ```bash
   git clone https://github.com/PaceKW/Haystack-RAG.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd Haystack-RAG
   ```

3. **Set Up a Virtual Environment:**

   To ensure that your project dependencies are isolated, it's recommended to use a virtual environment. Here’s how to set it up:

   - **Install `venv` (if not already installed):**
     Make sure you have Python installed. The `venv` module is included with Python 3.3 and later. You can check your Python version with:
     ```bash
     python --version
     ```

   - **Create a virtual environment:**
     Run the following command in your project directory:
     ```bash
     python -m venv .venv
     ```

   - **Activate the virtual environment:**
     - On **Windows:**
       ```bash
       .venv\Scripts\activate
       ```
     - On **macOS and Linux:**
       ```bash
       source .venv/bin/activate
       ```

4. **Create a `.env` file in the root directory and add your Groq API key:**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Install dependencies:**
   After activating the virtual environment, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the application:**
   With the virtual environment activated, you can run your application:
   ```bash
   python app.py
   ```

7. **Access the application in your browser:**
   Open `http://localhost:5000` in your web browser.

8. **Deactivate the virtual environment:**
   When you are done working, you can deactivate the virtual environment by running:
   ```bash
   deactivate
   ```

## Contributing

We welcome contributions! If you'd like to help improve Haystack-RAG, please feel free to create a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Repository Link

You can find the project repository at [Haystack-RAG GitHub Repository](https://github.com/PaceKW/Haystack-RAG.git).

