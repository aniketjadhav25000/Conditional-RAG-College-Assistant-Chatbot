```bash
# Conditional RAG College Assistant Chatbot

An AI-powered college chatbot built using LangGraph and Conditional RAG to answer academic, fee-related, and general questions.

## Overview

The chatbot classifies student queries and routes them to the appropriate processing path. It retrieves relevant information from college PDF documents when needed and generates answers based on the student's selected programme.

## Features

- LLM-based query classification
- Conditional routing using LangGraph
- Document retrieval using FAISS
- Programme-specific responses
- General question support without document retrieval
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
- python-dotenv
- Streamlit

## Project Workflow

1. The student selects a programme and enters a question.
2. The LLM classifies the query as academic, fee-related, or general.
3. LangGraph routes the query to the appropriate branch.
4. Relevant information is retrieved from the corresponding PDF when required.
5. The LLM generates the final response.

## Knowledge Sources

- `academics_handbook.pdf` — Academic rules and course information.
- `fee_structure.pdf` — Fee details and payment rules.

## Installation

### 1. Clone the Repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd conditional-rag-college-assistant

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Configure Environment Variables

Create a `.env` file and add your Groq API key:

GROQ_API_KEY=your_groq_api_key

Keep your API key private and never commit your `.env` file.

### 4. Run the Application

For the command-line application:

python app.py

For the Streamlit interface:

streamlit run streamlit_app.py

Replace the filenames with your actual filenames if they differ.

## Example Questions

- What are the attendance requirements?
- What is the fee structure for BCA?
- What are the examination rules?
- What is artificial intelligence?

## Future Improvements

- Add conversation memory.
- Improve document retrieval and answer accuracy.
- Include source citations in responses.
- Deploy the application online.

## Author

Aniket Jadhav

GitHub: YOUR_GITHUB_PROFILE_URL
```
