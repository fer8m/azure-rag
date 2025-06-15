import os
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd


app = FastAPI()

# openai.api_base = os.getenv("OPENAI_API_BASE")  # Your Azure OpenAI resource's endpoint value.
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_type = "azure"
# openai.api_version = "2023-05-15" 
client = OpenAI(
    base_url = os.getenv("OPENAI_API_BASE"),
    api_key = os.getenv("OPENAI_API_KEY")
)


embeddings = OpenAIEmbeddings(deployment="demo-embedding", chunk_size=1)

# Connect to Azure Cognitive Search
# acs = AzureSearch(azure_search_endpoint=os.getenv('SEARCH_SERVICE_NAME'),
#                  azure_search_key=os.getenv('SEARCH_API_KEY'),
#                  index_name=os.getenv('SEARCH_INDEX_NAME'),
#                  embedding_function=embeddings.embed_query)
encoder = SentenceTransformer('all-MiniLM-L6-v2') # Model to create embeddings
COLLECTION_NAME = "wine_files"

def init_rag(encoder: SentenceTransformer) -> QdrantClient:
    df = pd.read_csv('wine-ratings.csv')
    data = df.sample(n=200).to_dict('records')

    qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance
    
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
            distance=models.Distance.COSINE
        )
    )
    # vectorize!
    qdrant.upload_points(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=idx,
                vector=encoder.encode(doc["notes"]).tolist(),
                payload=doc,
            ) for idx, doc in enumerate(data) # data is the variable holding all the arxiv files
        ]
    )
    return qdrant

qdrant = init_rag(encoder)

class Body(BaseModel):
    query: str


@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
def ask(body: Body):
    """
    Use the query parameter to interact with the Azure OpenAI Service
    using the Azure Cognitive Search API for Retrieval Augmented Generation.
    """
    print(f"received: {body.query}")
    search_result = search(body.query)
    print(f"rag result: {search_result}")
    chat_bot_response = assistant(body.query, search_result)
    print(f"chat response: {chat_bot_response}")
    return {'response': chat_bot_response}



def search(query):
    """
    Send the query to Azure Cognitive Search and return the top result
    """
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=encoder.encode(query).tolist(),
        limit=5
    )
    search_results = [hit.payload for hit in hits]
    # print("Vector database result:")
    # for hit in search_results:
    #     print(hit)
    # print()
    return search_results


def assistant(query, context):
    completion = client.chat.completions.create(
        model="LLaMA_CPP",
        messages=[
            {"role": "system", "content": "Asisstant is a chatbot that helps you find the best wine for your taste."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": str(context)}
        ]
    )
    print("LLM response:")
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content