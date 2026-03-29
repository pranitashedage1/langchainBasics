# embedding means converting text into vector representation. 
# This is done by using a pre-trained model that has been trained on a large corpus of text data. 
# The model takes in a piece of text and outputs a vector that represents the meaning of the text. 
# This vector can then be used for various tasks such as similarity search, clustering, and classification.
# For this I will need opeAI Key

#  

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)

texts = [
    "Apple make good computers",
    "I believe Apple is innovative",
    "I love apples",
    "I am fan of MacBooks",
    "I enjoy oragnes",
    "I like Lenovo ThinkPads",
    "I think pears taste very good"
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search("Linux is a great operating system", k=7))


