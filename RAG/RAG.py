# openai api key를 env 파일에서 불러오기 
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------------
from datasets import load_dataset
from langchain_classic.docstore.document import Document

# KorQuAD 1.0 학습 데이터셋 불러오기
datasets = load_dataset('squad_kor_v1', split='train')
sample = datasets[100]

documents = []

id = sample["id"]
title = sample["title"]
context = sample["context"]

documents.append(
    Document(
        page_content=context, 
        metadata={
            "title" : title, 
            "id" : id
        }
    )
)

# ---------------------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 청크 분할
chunk_size = 200
overlap_ratio = 0.2
chunk_overlap = int(chunk_size * overlap_ratio)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"문서를 {len(chunks)}개의 청크로 분할했습니다. ")

# ---------------------------------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 임베딩 모델 초기화
embedding_model = HuggingFaceEmbeddings(
    model_name = "jhgan/ko-sroberta-multitask",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': True}
)

# 벡터 데이터베이스 생성
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

# ---------------------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=api_key
)

# 검색기 생성
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 프롬프트 생성
prompt = ChatPromptTemplate.from_template("""
You are an assistant for answering questions based on the given context.

Context:
{context}

Question:
{question}

Answer:
""")

def combine_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def format_prompt(x):
    return prompt.format(**x)

# RAG 체인 생성
rag_chain = (
    {
        "context": retriever | RunnableLambda(combine_docs),
        "question": RunnablePassthrough()
    }
    | RunnableLambda(format_prompt)
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------------------------------
# 샘플 테스트 
question = sample["question"]
real_answer = sample["answers"]["text"][0]

print("질문: ", question)
print("정답: ", real_answer)
print("모델 답: ", rag_chain.invoke(question))