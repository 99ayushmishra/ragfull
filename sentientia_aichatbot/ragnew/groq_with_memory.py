import os
from flask import Flask, render_template, request, redirect, url_for
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = 'uploads'

db = None
chat_history = []

@app.route('/', methods=['GET'])
def upload_file():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    global db
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    if not file.filename.endswith('.pdf'):
        return 'Only PDF files are allowed', 400

    # Save the file to the uploads folder
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # Load PDF document
    loader = PyPDFLoader(filename)
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunk_documents = text_splitter.split_documents(docs)

    # Create FAISS index with Ollama embeddings
    db = FAISS.from_documents(chunk_documents, OllamaEmbeddings(model='nomic-embed-text'))

    return redirect(url_for('ask_question'))


@app.route('/ask', methods=['GET'])
def ask_question():
    return render_template('ask.html', chat_history=chat_history)

@app.route('/answer', methods=['POST'])
def handle_question_groq():
    global chat_history
    question = request.form['question']

    if db is None:
        return 'No PDF uploaded yet', 400

    # Construct the chat history string
    chat_history_str = "\n".join([f"User: {entry['question']}\nBot: {entry['answer']}" for entry in chat_history])

    # Updated prompt template
    prompt = ChatPromptTemplate.from_template("""
    Given the following chat history and question, use the provided context to answer. Think step-by-step before providing a detailed answer.

    Chat History:
    {chat_history_str}

    <context> {context} </context>
    Question: {input}
    """)

    # Groq LLM
    groq_api_key = "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" 
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retriever
    retriever = db.as_retriever()

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Perform similarity search and get the answer
    retrieved_result = db.similarity_search(question)

    response = retrieval_chain.invoke({"input": question, "chat_history_str": chat_history_str})

    chat_history.append({"question": question, "answer": response['answer']})
    if len(chat_history) > 10:
        chat_history.pop(0)

    # Format the retrieved results
    retrieved_results_str = "\n".join([doc.page_content for doc in retrieved_result])

    return render_template('ask.html', question=question, answer=response['answer'], chat_history=chat_history, retrieved_results=retrieved_results_str)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)