# ==============================================================================
#  Step 1: Import Necessary Libraries
# ==============================================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# ==============================================================================
#  Step 2: Initialize the RAG Pipeline
# ==============================================================================
def initialize_rag_pipeline():
    """
    Initializes the RAG pipeline by loading templates and setting up the vector store.
    """
    print("Initializing RAG pipeline...")
    # This prevents a fork-related warning when running with Flask's reloader
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    loader = DirectoryLoader('templates/', glob="**/*.txt", show_progress=True)
    documents = loader.load()

    if not documents:
        print("Error: No document templates found in the 'templates' directory.")
        # Raise an error to stop the app if no templates are available
        raise FileNotFoundError("No document templates found in the 'templates' directory.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(texts, embeddings)
    print("RAG pipeline initialized successfully!")
    return vector_store.as_retriever(search_kwargs={"k": 3})

# ==============================================================================
#  Step 3: Set up the Flask App
# ==============================================================================
app = Flask(__name__)
# Enable CORS to allow requests from your frontend
CORS(app)

retriever = initialize_rag_pipeline()

# --- IMPORTANT ---
# For deployment, it's best practice to load this key from an environment variable.
# Example: google_api_key = os.getenv("GOOGLE_API_KEY")
google_api_key = "AIzaSyBgO9W4FUknwDg0DFNBdDxXSKXGTQo_9iI"

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set!")

# Use a known working and powerful model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key)

# ==============================================================================
#  Step 4: Define the Document Generation Logic
# ==============================================================================
@app.route('/generate-document', methods=['POST'])
def generate_document():
    try:
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "Invalid input. Please send a JSON body."}), 400

        context_parts = [f"{key.replace('_', ' ').title()}: {value}" for key, value in user_data.items()]
        user_context = "\n".join(context_parts)
        retrieved_docs = retriever.invoke(user_context)
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

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
#  Step 5: Run the Flask App with Production-Ready Settings
# ==============================================================================
if __name__ == '__main__':
    # Render will set the PORT environment variable.
    # We default to 5000 for local development.
    port = int(os.environ.get('PORT', 5000))
    
    # The host '0.0.0.0' makes the server publicly available.
    # use_reloader=False prevents a known conflict with the tokenizers library.
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

