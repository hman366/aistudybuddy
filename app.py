import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from PyPDF2 import PdfReader

# Template for custom prompt
template = """ 
You are a tutor helping me study for my exam using the provided context. 
{query}
"""

# Initializing the prompt
prompt = PromptTemplate.from_template(template)

# Function to get text vectors
def get_vectors(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Function to read text from PDF
def get_pdf_text(doc):
    text = ""
    pdf_reader = PdfReader(doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to process user query
def process_query(query):
    """Processes the query:
    1- appends the query to the prompt
    2- retrieves a response using the conversation object
    3- updates the chat history
    4- displays the chat history"""
    question = str(prompt.format(query=query))
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]

    for message in st.session_state.chat_history:
        if message.content.startswith("Helpful Answer:"):
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            break

# Function to create conversation chain
def get_conv(vects):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token="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",
        model_kwargs={"temperature": 0.7, "max_length": 2048},
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vects.as_retriever(), memory=memory
    )
    return conversation_chain

# Main function
def main():
    st.set_page_config(page_title="AI 6.0 Tutor", page_icon="🤖")
    st.header("AI Study Buddy 🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    query = st.text_input(
        """3 - Ask the study buddy to help you learn from your document:
    \nExample: "Give me a question that could figure on my final exam." """
    )
    if query:
        process_query(query)

    with st.sidebar:
        st.markdown(
            """
        # AI study buddy: helps you study buddy helps you study for your exams using your own course material:
        """
        )
        st.subheader("1 - Upload your document and hit 'Process'")
        st.markdown(
            """
        Example: [All about giraffe's](https://giraffeconservation.org/wp-content/uploads/2016/02/GCF-Giraffe-booklet-2017-LR-spreads-c-GCF.compressed.pdf)
        """
        )
        doc = st.file_uploader(
            "Your Document here",
            type="pdf",
        )
        if st.button("Process"):
            with st.spinner("Processing file ..."):
                raw_text = get_pdf_text(doc)
                chunks = get_chunks(raw_text)
                vects = get_vectors(chunks)
                st.session_state.conversation = get_conv(vects)
                st.write("2 - File Processed!, You can start learning!")

if __name__ == "__main__":
    main()
