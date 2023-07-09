from dotenv import load_dotenv
from langchain.vectorstores import FAISS
import torch
import os
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_PATH = os.getenv("OPENAI_EMBEDDING_MODEL_PATH")  # Path to the .pt model file
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")

def ask_question(qa, question):
    result = qa({"query": question})
    print("Question:", question)
    print("Answer:", result["result"])

if __name__ == "__main__":
    # load the OpenAI GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # load the faiss vector store we saved into memory
    vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", model)

    # use the faiss vector store we saved to search the local document
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})

    # use the vector store as a retriever
    qa = load_qa_chain(llm_model=model, tokenizer=tokenizer, retriever=retriever)
    
    while True:
        query = input('you: ')
        if query == 'q':
            break
        ask_question(qa, query)
