# ==============================================================================
#  Step 1: Import Necessary Libraries
# ==============================================================================
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
# Updated import for PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

# ==============================================================================
#  Step 2: Initialize the RAG Pipeline
# ==============================================================================
def initialize_rag_pipeline():
    """
    This function initializes the RAG pipeline by loading templates,
    creating embeddings, and setting up the vector store.
    """
    print("Initializing RAG pipeline...")

    # Load legal document templates from the 'templates' directory
    # Note: Using a loader for .txt files here for simplicity.
    loader = DirectoryLoader('templates/', glob="**/*.txt")
    documents = loader.load()

    # Check if documents were loaded
    if not documents:
        print("Warning: No document templates found in the 'templates' directory.")
        print("Please create a 'templates' folder and add .txt files (e.g., nda.txt).")
        # To prevent crashing, we'll exit if no templates are found.
        # In a real app, you might handle this more gracefully.
        exit()


    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings using a sentence transformer model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a Chroma vector store to hold the document embeddings
    vector_store = Chroma.from_documents(texts, embeddings)
    
    print("RAG pipeline initialized successfully!")
    # Return a retriever object that can find relevant documents
    return vector_store.as_retriever(search_kwargs={"k": 2})

# ==============================================================================
#  Step 3: Set up the Flask App
# ==============================================================================
app = Flask(__name__)
retriever = initialize_rag_pipeline()

# Configure the Gemini API key
# IMPORTANT: Replace "YOUR_GOOGLE_API_KEY" with your actual key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBgO9W4FUknwDg0DFNBdDxXSKXGTQo_9iI"

# *** FIX: Updated the model name to the correct version ***
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# ==============================================================================
#  Step 4: Define the Document Generation Logic
# ==============================================================================
@app.route('/generate-document', methods=['POST'])
def generate_document():
    """
    API endpoint to generate a legal document.
    Expects a JSON payload with user inputs.
    """
    try:
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "Invalid input. Please send a JSON body."}), 400

        # Create a detailed context string from the user's JSON data
        context_parts = [f"{key.replace('_', ' ').title()}: {value}" for key, value in user_data.items()]
        user_context = "\n".join(context_parts)

        # Retrieve relevant legal clauses using the modern 'invoke' method
        retrieved_docs = retriever.invoke(user_context)
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Create a prompt for the language model
        prompt_template = """
        You are an expert legal assistant. Your task is to draft a professional and accurate {document_type}.
        
        Use the following retrieved legal clauses and user-provided details to draft the document.
        Ensure that the document is well-structured, coherent, and legally sound. Fill in the user details in the appropriate places.

        **Retrieved Legal Clauses:**
        {retrieved_context}

        **User-Provided Details:**
        {user_context}

        **Generated {document_type}:**
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["document_type", "retrieved_context", "user_context"]
        )

        # Create the generation chain and invoke it
        chain = prompt | llm
        response = chain.invoke({
            "document_type": user_data.get("document_type", "Legal Document"),
            "retrieved_context": retrieved_context,
            "user_context": user_context
        })

        return jsonify({"generated_document": response.content})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to generate document"}), 500

# ==============================================================================
#  Step 5: Run the Flask App
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)

