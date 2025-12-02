# openai api key를 env 파일에서 불러오기 
import os
import time
from dotenv import load_dotenv
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------------
from datasets import load_dataset
from langchain_classic.docstore.document import Document

# 학습 데이터셋 불러오기 - 둘 중 선택 
datasets = load_dataset("klue", "mrc", split='train')
#datasets = load_dataset('squad_kor_v1', split='train')

documents = []
N = 300

for i in range(N): 
    item = datasets[i]
    title = item["title"]
    context = item["context"]

    documents.append(
        Document(
            page_content=context, 
            metadata={
                "title" : title, 
            }
        )
    )

# ---------------------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 청크 분할
results = []

chunk_size = [128, 256, 512]
overlap_ratio = [0, 0.1, 0.2]

for CS in chunk_size: 
    for OL in overlap_ratio:
        chunk_overlap = int(CS * OL) 
        print(f"\n\n Testing CHUNK_SIZE={CS}, OVERLAP_RATIO={OL} ... ")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CS,
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
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=api_key
        )

        # 검색기 생성
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant for answering questions in based on the given context.
        Answer the question in a couple of words, not in a sentence. 

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
        import re 

        # EM / f1 score 평가 지표 계산
        def normalize_text(text): 
            text = text.lower()
            text = re.sub(r"[^가-힣a-z0-9\s]", "", text)
            text = text.strip()
            return text 

        def exact_match(gold, pred): 
            return normalize_text(gold) == normalize_text(pred)

        def f1_score(gold, pred): 
            pred_tokens = normalize_text(pred).split()
            gold_tokens = normalize_text(gold).split()
            common = set(pred_tokens) & set(gold_tokens)

            if len(common) == 0: 
                return 0
            
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gold_tokens)
            return (2 * precision * recall) / (precision + recall)

        # 평균 EM/ f1 score 계산
        em_total = 0
        f1_total = 0

        for i in range(N): 
            sample = datasets[i]
            question = sample["question"]
            gold = sample["answers"]["text"][0]
            pred = rag_chain.invoke(question)

            if exact_match(gold, pred): 
                em_total += 1
            
            f1_total += f1_score(gold, pred)

        EM = em_total / N
        F1 = f1_total / N

        results.append((CS, OL, EM, F1))
        time.sleep(0.7)

# 세트별 평가 지표 결과 출력 
print("\n\nchunk size | overlap |  EM  | F1")
for cs, ol, em, f1 in results: 
    print(f"{cs:10d} | {ol:7.1f} | {em:.3f} | {f1:.3f}")