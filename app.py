# ==============================================================================
#  Step 1: Import Necessary Libraries
# ==============================================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
import threading

# ==============================================================================
#  Step 2: Lazy Initialization Setup
# ==============================================================================
init_lock = threading.Lock()
retriever = None
llm = None

def initialize_rag_pipeline():
    """
    This function initializes the entire RAG pipeline.
    It's designed to be called only once.
    """
    print("--- [START] RAG Pipeline Initialization (First Request) ---")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Step 1: Loading documents...")
    loader = DirectoryLoader('templates/', glob="**/*.txt", show_progress=True)
    documents = loader.load()
    if not documents:
        raise FileNotFoundError("Critical Error: No document templates found in 'templates/' directory.")
    print(f"Step 1 SUCCESS: Loaded {len(documents)} document(s).")

    print("Step 2: Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Step 2 SUCCESS: Split into {len(texts)} chunks.")

    print("Step 3: Configuring Google Gemini API...")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY environment variable not found!")
    print(f"Step 3 SUCCESS: GOOGLE_API_KEY loaded (ends with '...{google_api_key[-4:]}').")

    print("Step 4: Initializing Google's serverless embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    print("Step 4 SUCCESS: Embeddings model initialized.")

    print("Step 5: Creating Chroma vector store...")
    vector_store = Chroma.from_documents(texts, embeddings)
    print("Step 5 SUCCESS: Vector store created.")
    
    print("Step 6: Initializing Gemini LLM...")
    initialized_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key)
    print("Step 6 SUCCESS: Gemini LLM initialized.")
    
    print("--- [SUCCESS] RAG Pipeline Initialized ---")
    return vector_store.as_ retriever(search_kwargs={"k": 3}), initialized_llm

def get_pipeline():
    """
    This function ensures the RAG pipeline is initialized and returns the components.
    It uses a lock to be thread-safe.
    """
    global retriever, llm
    with init_lock:
        if retriever is None or llm is None:
            retriever, llm = initialize_rag_pipeline()
    return retriever, llm

# ==============================================================================
#  Step 3: Set up the Flask App
# ==============================================================================
app = Flask(__name__)

# *** THE FIX: Explicitly configure CORS for your live frontend URL ***
# This tells your backend that it's okay to accept requests from your frontend.
CORS(app, resources={r"/generate-document": {"origins": "https://nyaay-ai-frontend.onrender.com"}})


# ==============================================================================
#  Step 4: Define the Document Generation Logic
# ==============================================================================
@app.route('/generate-document', methods=['POST'])
def generate_document():
    try:
        current_retriever, current_llm = get_pipeline()
        
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "Invalid input. Please send a JSON body."}), 400

        context_parts = [f"{key.replace('_', ' ').title()}: {value}" for key, value in user_data.items()]
        user_context = "\n".join(context_parts)
        retrieved_docs = current_retriever.invoke(user_context)
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt_template = """
        You are an expert legal assistant. Your task is to draft a professional and accurate {document_type}.
        Use the retrieved legal clauses and user-provided details to draft the document.
        Fill in the user details in the appropriate places.
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

        chain = prompt | current_llm
        response = chain.invoke({
            "document_type": user_data.get("document_type", "Legal Document"),
            "retrieved_context": retrieved_context,
            "user_context": user_context
        })

        return jsonify({"generated_document": response.content})

    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return jsonify({"error": "Failed to generate document"}), 500

# Add a simple health check endpoint that Render can use
@app.route('/health')
def health_check():
    return "OK", 200

# ==============================================================================
#  Step 5: Run the Flask App (for local development)
# ==============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)

