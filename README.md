cat > README.md <<'EOF'
# Conditional RAG College Assistant Chatbot

A college assistant chatbot built using **LangGraph, LangChain, and Retrieval-Augmented Generation (RAG)**. It uses a conditional workflow to route student queries into academic, fee-related, or general categories and retrieves relevant information from college documents when required.

## Overview

The chatbot provides programme-specific answers by understanding the user's query and selecting the appropriate processing path.

It supports three programmes:
- BCA
- BBA
- B.Com (H)

## Key Features

- Conditional query routing using LangGraph.
- Academic and fee-related document retrieval.
- RAG-based responses using FAISS vector search.
- LLM-powered responses using Groq.
- Programme-specific query handling.
- Streamlit-based user interface.

## How It Works

| Query Category | Processing Method | Knowledge Source |
|---|---|---|
| Academic | RAG | Academics Handbook |
| Fee-related | RAG | Fee Structure |
| General | Direct LLM Response | General Knowledge |

## Conditional Workflow

The chatbot uses LangGraph conditional edges to classify student queries and route them to the appropriate processing path.

    User Query
         |
         v
    Query Router
         |
    +----+----+
    |    |    |
    v    v    v
 Academic Fees General
    |    |    |
    v    v    v
 Academic Fee RAG Direct LLM
   RAG   |       |
    |    |       |
    +----+-------+
         |
         v
    Final Answer

- **Query Router:** Identifies the category of the user's query.
- **Conditional Edges:** Direct the workflow to the selected node.
- **RAG Nodes:** Retrieve relevant information from the corresponding PDF.
- **General Node:** Handles queries that do not require document retrieval.
- **Final Answer:** Returns the generated response to the user.

## RAG Pipeline

For academic and fee-related queries, the application retrieves relevant information from the corresponding PDF documents.

    PDF Documents
         |
         v
    Document Loader
         |
         v
    Text Splitting
         |
         v
    Hugging Face Embeddings
         |
         v
    FAISS Vector Store
         |
         v
    Similarity Search
         |
         v
    Relevant Document Chunks
         |
         v
    Groq LLM
         |
         v
    Final Response

**RAG Process:**

1. **Document Loading:** Loads the academic handbook and fee structure PDFs.
2. **Text Splitting:** Divides documents into smaller chunks.
3. **Embeddings:** Converts text chunks into numerical vectors using Hugging Face.
4. **Vector Storage:** Stores the embeddings in a FAISS vector database.
5. **Retrieval:** Finds the most relevant chunks based on the user's query.
6. **Response Generation:** Uses the retrieved context to generate an answer through the Groq LLM.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangGraph | Conditional workflow and routing |
| LangChain | LLM and RAG integration |
| Groq | LLM inference |
| Hugging Face | Text embeddings |
| FAISS | Vector database |
| PyPDFLoader | PDF document loading |
| RecursiveCharacterTextSplitter | Text chunking |
| Streamlit | User interface |
| python-dotenv | Environment variable management |

## Project Structure

    Conditional-RAG-College-Assistant-Chatbot/
    |
    |-- app.py
    |-- conditional_RAG.py
    |-- academics_handbook.pdf
    |-- fee_structure.pdf
    |-- requirements.txt
    |-- README.md
    |-- .env.example
    |-- .gitignore
    |-- .dockerignore

Note: Create your local `.env` file for API credentials. Do not commit it to GitHub.

## Knowledge Sources

| Document | Description |
|---|---|
| academics_handbook.pdf | Academic information and programme details |
| fee_structure.pdf | Programme-wise fee information |

## Getting Started

### 1. Clone the Repository

    git clone https://github.com/aniketjadhav25000/Conditional-RAG-College-Assistant-Chatbot.git
    cd Conditional-RAG-College-Assistant-Chatbot

### 2. Create a Virtual Environment

    python -m venv venv

Activate it on Windows:

    venv\Scripts\activate

### 3. Install Dependencies

    pip install -r requirements.txt

### 4. Configure Environment Variables

Create a `.env` file in the project root:

    GROQ_API_KEY=your_groq_api_key

Replace `your_groq_api_key` with your actual Groq API key.

### 5. Run the Application

    streamlit run app.py

Open the local URL displayed in your terminal.

## Example Queries

- What are the eligibility criteria for BCA?
- Explain the examination rules.
- What is the fee structure for BBA?
- What are the admission requirements for B.Com?
- Tell me something about the college.

## Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Conditional Workflow | LangGraph conditional edges |
| Query Routing | Category-based query classification |
| Retrieval-Augmented Generation | Context-based answer generation |
| Embeddings | Hugging Face sentence embeddings |
| Vector Search | FAISS similarity search |
| LLM Integration | Groq API |

## Future Improvements

- Add more academic programmes.
- Support additional document formats.
- Improve retrieval accuracy.
- Add conversation history.
- Deploy the application online.

## Author

**Aniket Jadhav**

- GitHub: https://github.com/aniketjadhav25000
- Portfolio: https://portfolio-ani-react.netlify.app/
- LinkedIn: https://www.linkedin.com/in/aniket-jadhav-a72466236/

## License

This project is intended for educational and learning purposes.
EOF
