# Conditional RAG College Assistant Chatbot

A LangGraph-based AI assistant that uses conditional workflows to route queries and RAG to retrieve relevant information from documents and generate useful answers.

## Overview

The Conditional RAG College Assistant Chatbot helps students get answers to academic, fee-related, and general questions.

It uses LangGraph to classify questions and route them to the appropriate workflow. For academic and fee-related questions, it retrieves relevant information from college PDF documents. For general questions, it generates responses without document retrieval.

## Features

- Query classification using an LLM
- Conditional workflow routing using LangGraph
- Retrieval-Augmented Generation (RAG)
- PDF document loading and text splitting
- Semantic search using FAISS
- Hugging Face sentence embeddings
- Programme-specific responses
- Support for academic, fee-related, and general questions
- Interactive Streamlit interface

## Technologies Used

- Python
- LangGraph
- LangChain
- ChatGroq
- Hugging Face Embeddings
- FAISS
- PyPDFLoader
- RecursiveCharacterTextSplitter
- Streamlit
- python-dotenv

## Project Workflow

1. The student selects a programme and enters a question.
2. The LLM classifies the question as academic, fee-related, or general.
3. LangGraph uses conditional routing to select the appropriate workflow.
4. For academic questions, the system retrieves relevant information from the academic handbook.
5. For fee-related questions, the system retrieves relevant information from the fee structure document.
6. For general questions, the system responds without retrieving documents.
7. The LLM generates the final answer using the retrieved context or general knowledge.

## RAG Pipeline

1. Load college PDF documents using PyPDFLoader.
2. Split documents into smaller text chunks using RecursiveCharacterTextSplitter.
3. Convert the text chunks into vector embeddings using Hugging Face Embeddings.
4. Store the embeddings in a FAISS vector database.
5. Retrieve relevant document chunks based on the student's question.
6. Generate a response using the retrieved information and the LLM.

## Project Structure

<img width="497" height="292" alt="image" src="https://github.com/user-attachments/assets/d49d4296-175a-430b-b9fe-6d04e8fca28f" />


## Knowledge Sources

- academics_handbook.pdf — Contains academic rules and course information.
- fee_structure.pdf — Contains fee details and payment information.

## Installation and Setup

### 1. Clone the Repository
``` bash
git clone https://github.com/aniketjadhav25000/Conditional-RAG-College-Assistant-Chatbot.git

cd Conditional-RAG-College-Assistant-Chatbot
```
### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
```

Activate on Windows:
``` bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure Environment Variables

Create a .env file in the project directory and add your Groq API key:
```bash 
GROQ_API_KEY=your_groq_api_key
```
Replace your_groq_api_key with your actual API key.

Keep your API key private. Never upload your .env file or expose your API key in a public repository.

### 5. Run the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```
If your Streamlit entry point uses a different filename, replace app.py with the correct filename.

## Example Questions

- What are the attendance requirements?
- What are the examination rules?
- What is the fee structure for BCA?
- What academic rules should students follow?
- What courses are available for my programme?
- What is Artificial Intelligence?

## Key Concepts Demonstrated

- LangGraph: Builds and manages the application workflow.
- Conditional Workflow: Routes questions to different processing paths based on their classification.
- RAG: Retrieves relevant document information before generating an answer.
- Embeddings: Convert text into numerical representations for semantic search.
- FAISS: Stores and searches document vectors efficiently.
- LLM: Classifies questions and generates responses.

## Future Improvements

- Add conversation memory.
- Include document source citations in responses.
- Improve document retrieval accuracy.
- Add support for more college documents.
- Deploy the application online.

## Author

Aniket Jadhav

GitHub: https://github.com/aniketjadhav25000

## Repository

https://github.com/aniketjadhav25000/Conditional-RAG-College-Assistant-Chatbot

## License

This project is intended for educational and learning purposes.
