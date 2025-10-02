import streamlit as st
import ollama
import time
from collections import deque

import os
import shutil
import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter





# 사이드바 (PDF 데이터 확인으로 위로 수정)
st.sidebar.title("챗봇")
st.sidebar.write("버전 1.1")

# RAG 백엔드
@st.cache_resource
def setup_rag_system(folder_path, collection_name):
    """PDF를 로드하고 ChromaDB에 임베딩하는 RAG 시스템을 설정합니다."""
     # chroma_db 폴더 삭제
    # folder_delete = "./chroma_db"
    # if os.path.exists(folder_delete):
    #     shutil.rmtree(folder_delete)
    # else:
    #     st.info(f"'{folder_delete}' 폴더가 존재하지 않습니다.")

    # PDF 로드 및 청킹
    def load_and_split_pdfs(folder_path):
        all_chunks = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                st.info(f"PDF 파일 로드 중: {filename}")
                st.sidebar.write(f"{filename}")
                try:
                    pdf_data = pymupdf4llm.to_markdown(file_path)
                    text = "".join(pdf_data)
                    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n",".","!","?"], chunk_size=1000, chunk_overlap=150)
                    chunks = text_splitter.split_text(text)
                    all_chunks.extend(chunks)
                except Exception as e:
                    st.error(f"오류 발생: {filename}, {e}")
                    continue
        return all_chunks

    # 임베딩 및 벡터 DB 저장
    with st.spinner("환경 구성 중..."):
        embedder = SentenceTransformer("intfloat/multilingual-e5-small")

        chroma_db_path = "./chroma_db"
        if os.path.exists(chroma_db_path):
            print(f"기존 ChromaDB 폴더({chroma_db_path}) 삭제 중...")
            shutil.rmtree(chroma_db_path)
            print("삭제 완료.")


        client = chromadb.PersistentClient(path="./chroma_db")

        try:
            client.delete_collection(collection_name)
            st.success("기존 컬렉션을 삭제했습니다.")
        except chromadb.errors.NotFoundError:
            # st.info("삭제할 기존 컬렉션이 없습니다.")
            print("삭제할 기존 컬렉션이 없습니다.")

        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        st.info("문서 로드 및 임베딩 시작...")
        start_time = time.time()
        chunks = load_and_split_pdfs(folder_path)
        
        if chunks:
            embeddings = embedder.encode(chunks, convert_to_tensor=False)
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                collection.add(
                    ids=[f"chunk_{i + 1}"],
                    embeddings=[embedding.tolist()],
                    metadatas=[{"text": chunk}],
                )
            st.success(f"문서 준비 완료! 소요 시간: {time.time() - start_time:.2f}초")
        else:
            st.warning("지정된 폴더에 PDF 파일이 없습니다.")
    
    return embedder, collection

# 질문 처리 함수
def retrieve_docs(query, collection, embedder, top_k=2):
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    if not results["metadatas"]:
        return ["관련 문서를 찾을 수 없습니다."]
    
    docs = [doc["text"] for doc in results["metadatas"][0]]
    return docs



# 메세지 매니저 클래스 구현
class Message_manager:
    def __init__(self):
        self._system_msg = {"role": "system", "content": ""}
        self.queue = deque(maxlen=1)  # 최대 1개 대화 저장 

    def create_msg(self, role, content):
        return {"role": role, "content": content}

    def system_msg(self, content):
        self._system_msg = self.create_msg("system", content)

    def append_msg(self, content):
        msg = self.create_msg("user", content)
        self.queue.append(msg)

    def generate_prompt(self, retrieved_docs):
        docs = "\n".join(retrieved_docs)
        prompt = [
            self._system_msg,
            {
                "role": "system",
                "content": f"문서 내용: {docs}\n질문에 대한 답변은 문서 내용을 기반으로 정확히 제공하시오.",
            },
        ] + list(self.queue)
        return prompt
    
msgManager = Message_manager()


# 시스템 메세지 등록
msgManager.system_msg(
    "당신은 유능한 챗봇입니다. 사용자의 질문에 대해 친절하고 정확하게 답변해주세요."
)

# AI 응답 채팅창 ( streamlit )

model = "exaone3.5"
folder_path = ".\\docs"
collection_name = "rag_collection"

# `@st.cache_resource`로 RAG 시스템을 한 번만 초기화
try:
    embedder, collection = setup_rag_system(folder_path, collection_name)
except FileNotFoundError:
    st.error("`docs` 폴더를 찾을 수 없습니다. PDF 파일을 넣고 다시 실행해주세요.")
    st.stop()


# 세션 상태에 메시지 기록 및 메시지 매니저 저장
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_manager" not in st.session_state:
    st.session_state.msg_manager = Message_manager()



# 채팅창을 위한 스크롤 여백 추가
st.markdown("<br><br><br><br>", unsafe_allow_html=True)

# 채팅 기록이 없으면 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 채팅 기록을 화면에 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자가 새로운 메시지를 입력하면 아래 코드 실행
if prompt := st.chat_input("메시지를 입력하세요..."):
    st.empty() 
    # 사용자 메시지를 채팅 기록에 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 봇의 응답
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            start_time = time.time()
            retrieved_docs = retrieve_docs(prompt, collection, embedder)
            msgManager = st.session_state.msg_manager
            msgManager.append_msg(prompt)
            msg = msgManager.generate_prompt(retrieved_docs)
           
            # 스트리밍으로 답변 생성
            placeholder = st.empty()
            full_answer = f"봇 :"

            is_first_chunk = True
            start_answer_time = None

            for response in ollama.chat(model=model, messages=msg, stream=True):
                if is_first_chunk:
                    start_answer_time = time.time()
                    is_first_chunk = False
                    
                chunk = response["message"]["content"]
                full_answer += chunk
                placeholder.markdown(full_answer)
            
            # LLM 추론 시간과 답변 시작 시간 출력
            if start_answer_time:
                st.success(f"답변 시작 시간: {start_answer_time - start_time:.2f}초")

            st.success(f"[총 소요 시간: {time.time() - start_time:.2f}초]\n")

    # 봇의 응답을 채팅 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": full_answer})

