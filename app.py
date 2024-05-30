from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Загрузка модели Llama
ollama_llm = Ollama(model='llama3')

# Инициализация объектов
parser = StrOutputParser()
llama_chain = ollama_llm | parser

# Загрузка и обработка документа
loader = TextLoader('data1.txt', encoding='utf-8')
document = loader.load()
spliter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = spliter.split_documents(document)

# Создание векторного хранилища и поисковика на основе HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_storage = FAISS.from_documents(chunks, embedding_model)
retriever = vector_storage.as_retriever()

# Пример запроса к поисковику
# retriever.invoke('Инструкцию к какому прибору ты хранишь?')

# Создание шаблона запроса
template = """
You are AI-powered chatbot designed to provide 
information and assistance for customers
based on the context provided to you only. 

Context:{context}
Question:{question}
"""
prompt = PromptTemplate.from_template(template=template)
formatted_prompt = prompt.format(
    context=' Here is a context to use',
    question=' This is a question to answer'
)

# Создание цепочки запросов
result = RunnableParallel(context=retriever, question=RunnablePassthrough())
chain = result | prompt | ollama_llm | parser

# Примеры запросов к цепочке
# print(chain.invoke('общие сведения инструкции по эксплуатации'))
print(chain.invoke('что содержит твой контекст? отвечай на русском языке'))
