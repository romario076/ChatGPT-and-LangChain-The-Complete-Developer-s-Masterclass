from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="emb", embedding_function=embeddings)

#retriever = db.as_retriever()
retriever = RedundantFilterRetriever(embeddings=embeddings,chroma=db)

chain = RetrievalQA.from_chain_type(llm=chat,retriever=retriever,chain_type="stuff")

result = chain.invoke("What is an interesting fact about the English language?")

print(result)


