# importing libraries
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

# template for custom prompt, I found it gave better results
template = """ 
You are a tutor helping me study for my exam using the provided documents, use that to answer any questions that I have about the content. 
{query}
"""

# initializing the prompt
prompt = PromptTemplate.from_template(template)


def get_vectors(chunks):
    """
    computes the embeddings using pubmedbert-base-embeddings,
    uses FAISS to store them.
    """
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_pdf_text(doc):
    """Reads a pdf document and returns all of it as a string"""
    text = ""
    pdf_reader = PdfReader(doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_chunks(text):
    """splits the text into chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks




def process_query(query):
    """Processes the query:
    1- appends the query to the prompt
    2- retrieves a response using the conversation object
    3- updates the chat history
    4- displays the chat history"""
    question = str(prompt.format(query=query))
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:

            st.write(
            
                user_template.replace("{{MSG}}", message.content[141:]),
                unsafe_allow_html=True,
            )
        else:
            token = "Helpful Answer:"
            token_index = message.content.rfind(token)
            start_index = token_index + len(token)
            stripped = message.content[start_index:]

            st.write(
                bot_template.replace("{{MSG}}", stripped), unsafe_allow_html=True
            )


def get_conv(vects):
    """
    creating a conversation chain:
    """
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token="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",
        model_kwargs={"tempearture": 0.1, "max_length": 8192},
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vects.as_retriever(), memory=memory
    )
    return conversation_chain


def main():
    """
    main function running everything
    """
    st.set_page_config(page_title="AI Study Buddy  10.0", page_icon="🤖")
    st.header("AI study Buddy 🤖")
    st.write(css, unsafe_allow_html=True)

    # create session state object to use these variables outside of their scope
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # receiving user's query
    query = st.text_input(
        """3 - Ask the tutor to help you learn from your document:
    \nExample: "Give me a question that could show up on my final exam." """
    )
    if query:
        if st.session_state.conversation != None:
            process_query(query)

    with st.sidebar:
        # creating a sidebar that will contain an interface for the user to upload his document
        st.markdown(
            """
        # AI Study Buddy helps you study for your exams using your own course material:
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
            # processing the file
            with st.spinner("Processing file ..."):
                # get pdf files:
                raw_text = get_pdf_text(doc)
                # get chunks of texts
                chunks = get_chunks(raw_text)
                # get vectorstore
                vects = get_vectors(chunks)
                # get conversation
                st.session_state.conversation = get_conv(vects)
                st.write("2 - File Processed!, You can start learning!")


if __name__ == "__main__":
    main()
