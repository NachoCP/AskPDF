from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI


def get_qa_retriever(open_ai_key,
                     retriever):
    
    model = ChatOpenAI(api_key = open_ai_key)


    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])


    qa = RetrievalQA.from_chain_type(llm=model,
                                chain_type="stuff",
                                retriever=retriever,
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": prompt})
    return qa