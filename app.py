from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint


# 1. Load the local data file
loader = TextLoader("data.txt")
documents = loader.load()
# print("Loaded Documents:")
# print(documents)

# 2. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print("\n Split Documents:")
for i, doc in enumerate(docs[:3]):  # Just printing first 3 for brevity
    print(f"Chunk {i+1}:", doc.page_content)

# 3. Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# print("\n Embedding model loaded:", embedding_model)

# 4. Create vector store (FAISS)
vectorstore = FAISS.from_documents(docs, embedding_model)
print("\n Vector store created with", len(docs), "documents")

# 5. Create retriever
retriever = vectorstore.as_retriever()
print("\n Retriever is ready")

# 6. Load the LLM

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",  # explicitly define endpoint
    huggingfacehub_api_token="Your Token here",
    task="text2text-generation",
    temperature=0.5,
    model_kwargs={"max_length": 100}
)

print("\n LLM loaded:", llm)

# 7. Setup RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print("\n RAG chain is ready")


query = "What is this document about?"
result = qa_chain.invoke({"query": query})  

print("\n Question:", query)
print("💡 Answer:", result)
