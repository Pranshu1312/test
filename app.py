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
    Initializes the RAG pipeline with detailed logging for debugging.
    """
    print("--- [START] RAG Pipeline Initialization ---")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Step 1: Loading documents from 'templates/' directory...")
    loader = DirectoryLoader('templates/', glob="**/*.txt", show_progress=True)
    documents = loader.load()
    print(f"Step 1 SUCCESS: Loaded {len(documents)} document(s).")

    if not documents:
        raise FileNotFoundError("Critical Error: No document templates found in 'templates/' directory.")

    print("Step 2: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Step 2 SUCCESS: Split documents into {len(texts)} chunks.")

    print("Step 3: Initializing embeddings model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Step 3 SUCCESS: Embeddings model initialized.")

    print("Step 4: Creating Chroma vector store...")
    vector_store = Chroma.from_documents(texts, embeddings)
    print("Step 4 SUCCESS: Vector store created.")
    
    print("--- [SUCCESS] RAG Pipeline Initialized ---")
    return vector_store.as_retriever(search_kwargs={"k": 3})

# ==============================================================================
#  Step 3: Set up the Flask App
# ==============================================================================
print("--- [START] Application Setup ---")
app = Flask(__name__)
CORS(app)

retriever = initialize_rag_pipeline()

print("Step 5: Configuring Google Gemini API...")
google_api_key = os.environ.get("GOOGLE_API_KEY")

if not google_api_key:
    print("CRITICAL ERROR: GOOGLE_API_KEY environment variable not found!")
    raise ValueError("GOOGLE_API_KEY is not set!")
else:
    # Print a safe confirmation that the key was loaded
    print("Step 5 SUCCESS: GOOGLE_API_KEY loaded successfully (ends with '...{}').".format(google_api_key[-4:]))

print("Step 6: Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key)
print("Step 6 SUCCESS: Gemini LLM initialized.")

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

        chain = prompt | llm
        response = chain.invoke({
            "document_type": user_data.get("document_type", "Legal Document"),
            "retrieved_context": retrieved_context,
            "user_context": user_context
        })

        return jsonify({"generated_document": response.content})

    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return jsonify({"error": "Failed to generate document"}), 500

# ==============================================================================
#  Step 5: Run the Flask App
# ==============================================================================
if __name__ == '__main__':
    print("--- [START] Flask Server ---")
    # Render provides the PORT environment variable.
    port = int(os.environ.get('PORT', 10000))
    print(f"Attempting to bind to 0.0.0.0 on port {port}...")
    # Use debug=False for production environments
    app.run(host='0.0.0.0', port=port, debug=False)

