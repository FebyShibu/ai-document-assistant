
# AI Document Assistant

An AI-powered document assistant that allows users to upload PDF documents, ask questions in natural language, and receive context-aware answers with source citations.

The application uses Retrieval-Augmented Generation (RAG) and Azure OpenAI to retrieve relevant information from documents and generate grounded responses.

## Current Features

- Multiple PDF uploads
- PDF text extraction and overlapping text chunking
- Semantic search using vector embeddings
- Cosine similarity-based information retrieval
- AI-generated answers using Azure GPT-5-mini
- Source and page-level citations
- Context-aware follow-up questions
- Separate conversation history for individual documents
- Combined processing of multiple documents
- Interactive dark-themed chat interface
- Secure environment variable configuration

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive user interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Vector embeddings |
| Cosine Similarity | Semantic search |
| Azure OpenAI | AI response generation |
| GPT-5-mini | Large language model |
| Git and GitHub | Version control |

## Application Architecture

1. Users upload PDF documents.
2. Text is extracted from each document.
3. Extracted text is divided into overlapping chunks.
4. Chunks are converted into vector embeddings.
5. Relevant passages are retrieved using semantic search.
6. Retrieved context is passed to Azure OpenAI.
7. The AI generates an answer with source citations.

## Local Installation

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file using `.env.example` as a template.

Configure your own Azure OpenAI endpoint, deployment name and API key.

Run the application:

```bash
streamlit run app.py
```

## Security

Azure credentials are managed through environment variables.

Confidential credentials, personal documents and local database files are excluded from version control.

## Development Roadmap

- [x] PDF text extraction
- [x] Text chunking and vector embeddings
- [x] Semantic search
- [x] Azure OpenAI integration
- [x] Page-level citations
- [x] Multiple PDF uploads
- [x] Document-specific conversation history
- [x] Combined multi-document processing
- [ ] Cross-document question answering
- [ ] Enhanced source citations
- [ ] Improved user interface
- [ ] Azure SQL Database integration
- [ ] Persistent document and conversation storage
- [ ] Azure Blob Storage integration
- [ ] User authentication
- [ ] Azure cloud deployment
- [ ] RAG evaluation and automated testing

## Project Status

Currently under active development.

The objective is to build an end-to-end, cloud-based AI document management and question-answering application using modern AI engineering practices.
