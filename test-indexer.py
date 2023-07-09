from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import torch
import os
import io
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")

class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}  # Add an empty metadata dictionary

if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add a padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataPath = "./data/documentation/"
    fileName = dataPath + "azure-azure-functions.pdf"

    # Use PyPDF2 to load the PDF document
    with open(fileName, "rb") as f:
        pdf = PdfReader(f)
        pages = [page.extract_text() for page in pdf.pages]

    # Create Document objects for each page
    documents = [Document(page_content) for page_content in pages]

    # Create embeddings for each page
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)
    embeddings = embeddings.embed_documents(pages)

    # Use Langchain to create the embeddings
    texts = [document.page_content for document in documents]
    db = FAISS.from_texts(texts=texts, embedding=embeddings)

    # Save the embeddings into FAISS vector store
    db.save_local("./dbs/documentation/faiss_index")
