@@ -1,4 +1,3 @@
# importing libraries
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
@@ -11,71 +10,58 @@
from htmlTemplates import css, bot_template, user_template
from PyPDF2 import PdfReader


# template for custom prompt, I found it gave better results
# Template for custom prompt
template = """ 
You are a tutor helping me study for my exam using the provided context. 
{query}
"""

# initializing the prompt
# Initializing the prompt
prompt = PromptTemplate.from_template(template)


# Function to get text vectors
def get_vectors(chunks):
    """
    computes the embeddings using pubmedbert-base-embeddings,
    uses FAISS to store them.
    """
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


# Function to read text from PDF
def get_pdf_text(doc):
    """Reads a pdf document and returns all of it as a string"""
    text = ""
    pdf_reader = PdfReader(doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to split text into chunks
def get_chunks(text):
    """splits the text into chunks"""
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
    4- displays only the latest response without appending previous ones"""
    question = str(prompt.format(query=query))
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]

    # Display only the latest response from the AI without appending any previous responses
    latest_bot_response = st.session_state.chat_history[-1].content
    answer_start = latest_bot_response.find("Helpful Answer:") + len("Helpful Answer:")
    trimmed_response = latest_bot_response[answer_start:].strip()

    st.write(
        bot_template.replace("{{MSG}}", trimmed_response),
        unsafe_allow_html=True
    )



    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content[84:]),
                unsafe_allow_html=True,
            )
        else:
            # Modify to show only the response, not the previous messages
            if "Human:" not in message.content:
                st.write(
                    bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
                )

# Function to create conversation chain
def get_conv(vects):
    """
    creating a conversation chain:
    """
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token="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",
@@ -87,22 +73,17 @@ def get_conv(vects):
    )
    return conversation_chain


# Main function
def main():
    """
    main function running everything
    """
    st.set_page_config(page_title="AI 5.0 Tutor", page_icon="🤖")
    st.set_page_config(page_title="AI 6.0 Tutor", page_icon="🤖")
    st.header("AI Study Buddy 🤖")
    st.write(css, unsafe_allow_html=True)

    # create session state object to use these variables outside of their scope
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # receiving user's query
    query = st.text_input(
        """3 - Ask the study buddy to help you learn from your document:
    \nExample: "Give me a question that could figure on my final exam." """
@@ -111,10 +92,9 @@ def main():
        process_query(query)

    with st.sidebar:
        # creating a sidebar that will contain an interface for the user to upload his document
        st.markdown(
            """
        # HabibAI: helps you study buddy helps you study for your exams using your own course material:
        # AI study buddy: helps you study buddy helps you study for your exams using your own course material:
        """
        )
        st.subheader("1 - Upload your document and hit 'Process'")
@@ -128,18 +108,12 @@ def main():
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
