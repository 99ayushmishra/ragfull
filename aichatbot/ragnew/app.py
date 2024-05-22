import os
import sqlite3
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
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
app.secret_key = 'supersecretkey'  # Set a secret key for session management

db = None

def get_db_connection():
    conn = sqlite3.connect('chat.db')
    conn.row_factory = sqlite3.Row
    return conn

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

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    loader = PyPDFLoader(filename)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunk_documents = text_splitter.split_documents(docs)
    db = FAISS.from_documents(chunk_documents, OllamaEmbeddings(model='nomic-embed-text'))

    # Create a new session ID for the user
    session['session_id'] = str(uuid.uuid4())

    # Clear any existing chat history for this session
    conn = get_db_connection()
    conn.execute('DELETE FROM sessions WHERE session_id = ?', (session['session_id'],))
    conn.commit()
    conn.close()

    return redirect(url_for('ask_question'))

@app.route('/ask', methods=['GET'])
def ask_question():
    conn = get_db_connection()
    chat_history = conn.execute('SELECT * FROM sessions WHERE session_id = ?', (session['session_id'],)).fetchall()
    conn.close()
    return render_template('ask.html', chat_history=chat_history, session_id=session['session_id'])

@app.route('/answer', methods=['POST'])
def handle_question_groq():
    if db is None:
        return jsonify({"error": "No PDF uploaded yet"})

    question = request.form['question']
    session_id = session['session_id']

    conn = get_db_connection()
    chat_history = conn.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,)).fetchall()

    # Convert chat_history rows to a list of dictionaries
    chat_history_dicts = [dict(row) for row in chat_history]

    chat_history_str = "\n".join([f"User: {entry['question']}\nBot: {entry['answer']}" for entry in chat_history_dicts])

    prompt = ChatPromptTemplate.from_template("""
    Given the following chat history and question, use the provided context to answer. Think step-by-step before providing a detailed answer.

    Chat History:
    {chat_history_str}

    <context> {context} </context>
    Question: {input}
    """)

    groq_api_key = "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"
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    retrieved_result = db.similarity_search(question)
    response = retrieval_chain.invoke({"input": question, "chat_history_str": chat_history_str})

    answer = response['answer']

    # Store the question and answer in the database
    conn.execute('INSERT INTO sessions (session_id, question, answer) VALUES (?, ?, ?)',
                 (session_id, question, answer))
    conn.commit()
    conn.close()

    retrieved_results_str = "\n".join([doc.page_content for doc in retrieved_result])

    response_data = {
        "question": question,
        "answer": answer,
        "chat_history": chat_history_dicts,  # Send the chat history as a list of dictionaries
        "retrieved_results": retrieved_results_str,
    }

    return jsonify(response_data)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)