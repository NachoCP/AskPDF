# it will open in the localhost:8501
import streamlit as st
import os

from vector_store import get_vector_store
from llm import get_qa_retriever


st.set_page_config("Chat PDF")
st.header("Ask your PDF questions using LLama3")
st.subheader('File types supported: PDF')

temp_file = "./temp.pdf"
with st.sidebar:
    st.title("Menu:")
    password = st.text_input("OPEN AI Key",
                             type="password")
    st.session_state.open_ai_key = password

    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                type="pdf")
    if st.button("Submit & Process"):
        if "open_ai_key" not in st.session_state:
            st.error("Introduce Open AI Key")
        elif pdf_docs is None:
            st.error("Please upload a file before submitting anything")
        with st.spinner("Processing..."):
            st.write("File uploaded sucessfully")
            with open(temp_file, "wb") as file:
                file.write(pdf_docs.getvalue())
            
            retriever = get_vector_store(password, temp_file)
            chain = get_qa_retriever(password, retriever)
            st.session_state.chain = chain
            st.success("Initialize the QA Retriever")

if "chain" in st.session_state:
    
    chain = st.session_state.chain
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        history = [
                f"{message['role']}: {message['content']}"
                for message in st.session_state.messages
            ]
        
        chat_history = [(message.split(": ")[0], message.split(": ")[1]) for message in history]
        
        print(prompt)
        
        result = chain({
                "query" : prompt
            })
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            print(result)
            full_response = result["result"]
            
            pages = set(str(doc.metadata["page"]) for doc in result["source_documents"])
            pages_txt = f"There is more information in these pages: {', '.join(sorted(pages))}"
            full_response = f"{full_response} \n {pages_txt}"
            message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Please upload your PDF")