# Multimodal RAG for HealthCare Applications 

## Overview  
This project implements **Multimodal Retrieval-Augmented Generation (RAG)** for medical document analysis, leveraging text, tables, and images. The approach enables advanced question-answering by integrating structured and unstructured data sources.  

**Case Study:** Multimodal RAG was performed on the medical document:  
**[AHA/ACC/HFSA Clinical Practice Guideline: 2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure](https://www.ahajournals.org/doi/epdf/10.1161/CIR.0000000000001063)**.  

Inspired by the LangChain cookbook example:  
[LangChain Semi-structured & Multimodal RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb).  
[Multimodal RAG: Chat with PDFs (Images & Tables)](https://www.youtube.com/watch?v=uLrReyH5cu0), [langchain-multimodal](https://colab.research.google.com/gist/alejandro-ao/47db0b8b9d00b10a96ab42dd59d90b86/langchain-multimodal.ipynb#scrollTo=e4adfeba).

## **Real-World Use Cases** 🚀  

#### 🏥 **Clinical Decision Support**  
#### 📄 **Medical Research & Summarization**  
#### 🤖 **AI-Powered Healthcare Assistants** 
#### 🎓 **Medical Compliance & Training**  
#### 💊 **Treatment Plans & Drug Prescriptions**  

## Components  

### 1. **LLM Models**  
- **`llama-3.1-8b-instant`**: A fast, lightweight model for text-based retrieval and generation.  
- **`llama-3.2-11b-vision-preview`**: A multimodal model capable of processing text and medical images for advanced RAG tasks.  

### 2. **Document Parsing with Unstructured.io**  
- Extracts structured and unstructured data from PDFs, tables, and medical charts.  
- Enables processing of complex medical reports and guidelines.  

### 3. **MultiVectorRetriever from LangChain**  
- **Hybrid retrieval**: Stores embeddings of medical text and tables for improved search.  
- Enhances retrieval by considering both semantic similarity (vector store) and structured data (document store).  

### 4. **Vector Store + Document Store**  
- **Vector Store (e.g., FAISS, Chroma)**: Stores dense embeddings for medical text retrieval.  
- **Document Store (e.g., InMemoryStore)**: Keeps original medical reports for referencing after retrieval.  

### 5. **Table & Image Summarization**  
- **Table Summarization**: Extracts critical insights from structured tables in medical guidelines.  
- **Image Summarization**: Generates captions, extracts text (OCR), and analyzes medical images for context-aware responses.  

## Workflow  
1. **Document Ingestion**: Parse text, tables, and images from the **AHA/ACC/HFSA Guideline for the Management of Heart Failure**.  
2. **Embedding Generation**: Convert text and medical images into vector representations.  
3. **Hybrid Retrieval**: Use MultiVectorRetriever to retrieve relevant multimodal content.  
4. **Contextual Generation**: Feed retrieved content into `llama-3.2-11b-vision-preview` or some other multimodal model for medical response generation.  
5. **QA**: Generate responses with references to text, tables, and images from the medical document.  

This approach enables **intelligent medical question-answering, report summarization, and document analysis** by leveraging **structured and unstructured healthcare data**. 🚀  


## How to use:
-   Clone this repository `git clone <repository-url>`
-   Initialize poetry with `poetry init -n`
-   Run `poetry config virtualenvs.in-project true` so that virtualenv will be present in project directory
-   Run `poetry env use <C:\Users\username\AppData\Local\Programs\Python\Python311\python.exe>` to create virtualenv in project (change username to your username)
-   Run `poetry shell`
-   Run `poetry install` to install requried packages
-   Create `.env` file and insert all keys: see `.env.example` file
-   Use `multimodal_rag_langchain.ipynb` notebook to test multi modal rag
