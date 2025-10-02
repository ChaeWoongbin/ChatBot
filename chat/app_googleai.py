import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import pymupdf  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb # faiss 대신 chromadb 임포트
import shutil # shutil 라이브러리 추가

# .env 파일에서 환경 변수 로드
load_dotenv('api_key.env')

# --- 설정 및 초기화 ---

# Google API 키 설정
try:
    if "GEMINI_API_KEY" not in os.environ:
        st.error("GEMINI_API_KEY 환경 변수를 설정해주세요! (.env 파일)")
        st.stop()
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"API 키 설정 중 오류 발생: {e}")
    st.stop()

# --- ChromaDB 버전: docs 폴더에서 PDF 로드 및 처리 ---

@st.cache_resource
def load_and_process_pdfs_from_docs():
    """'./docs' 폴더에서 모든 PDF를 로드하고 ChromaDB에 저장합니다."""
    DOCS_PATH = "./docs"
    CHROMA_PATH = "./chroma_db" # ChromaDB 데이터를 저장할 폴더
    COLLECTION_NAME = "pdf_rag_collection"

    if not os.path.exists(DOCS_PATH):
        st.error(f"'{DOCS_PATH}' 폴더를 찾을 수 없습니다. PDF 파일을 넣어주세요.")
        st.stop()

    # 기존 ChromaDB 폴더가 있으면 삭제
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            # print(f"기존 ChromaDB 폴더 '{CHROMA_PATH}' 삭제 완료.")
        except Exception as e:
            st.warning(f"ChromaDB 폴더 삭제 실패: {e}. 수동으로 삭제하거나 권한을 확인해주세요.")
            # 삭제 실패 시 중단하지 않고 계속 진행하도록 합니다.

    # ChromaDB 클라이언트 초기화
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 기존 컬렉션이 있으면 삭제 후 새로 생성
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # 기존 컬렉션이 있다면 삭제 (항상 최신 docs 폴더 내용으로 다시 만들기 위함)
    # 더 발전된 방법: 파일 변경 여부를 체크하여 필요할 때만 업데이트
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=COLLECTION_NAME)

    # 새 컬렉션 생성
    collection = client.create_collection(name=COLLECTION_NAME)
    
    all_text = ""
    loaded_files = []
    
    with st.spinner("`docs` 폴더에서 PDF 파일을 읽는 중..."):
        for filename in os.listdir(DOCS_PATH):
            if filename.endswith(".pdf"):
                file_path = os.path.join(DOCS_PATH, filename)
                try:
                    with pymupdf.open(file_path) as doc:
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        all_text += text + "\n\n"
                        loaded_files.append(filename)
                except Exception as e:
                    st.warning(f"'{filename}' 파일 처리 중 오류 발생: {e}")
    
    if not loaded_files:
        st.error("`docs` 폴더에 처리할 PDF 파일이 없습니다.")
        st.stop()

    # 텍스트 청킹
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300,separators=[
                                                                                                    "\n\n",   # 단락
                                                                                                    "\n",     # 줄바꿈
                                                                                                    ".",      # 마침표 (문장)
                                                                                                    "?",      # 물음표
                                                                                                    "!",      # 느낌표
                                                                                                    " ",      # 공백 (단어)
                                                                                                    "",       # 문자
                                                                                                ])
    chunks = text_splitter.split_text(all_text)

     # =========================================================
    # 디버깅 코드 추가
    print("="*50)
    print(f"로드된 파일 개수: {len(loaded_files)}")
    print(f"추출된 전체 텍스트 길이: {len(all_text)}")
    print(f"생성된 청크 개수: {len(chunks)}")
    print("="*50)
    # =========================================================

    # 청크가 비어있는지 확인하는 방어 코드 추가
    if not chunks:
        st.error("PDF에서 텍스트를 추출하지 못했습니다. PDF가 이미지로만 구성되어 있는지 확인해주세요.")
        st.stop() # 프로그램 중단


    # 임베딩 및 ChromaDB에 저장
    try:
        with st.spinner("문서 임베딩 및 벡터 저장소에 저장 중..."):
            result = genai.embed_content(
                model="models/embedding-001",
                content=chunks,
                task_type="retrieval_document"
            )
            embeddings = result['embedding']
            
            # ChromaDB에 데이터 추가
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
            
            st.success("문서 처리가 완료되었습니다!")
            return collection, loaded_files
    except Exception as e:
        st.error(f"임베딩 또는 벡터 저장소 생성 중 오류 발생: {e}")
        st.stop()

# --- ChromaDB 버전: RAG 검색 함수 ---
def get_relevant_chunks(query, collection):
    """사용자 질문과 관련된 청크를 ChromaDB에서 검색합니다."""
    try:
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=6 # 상위 6개 결과
        )
        
        return results['documents'][0]
    except Exception as e:
        st.error(f"관련 문서 검색 중 오류 발생: {e}")
        return []

# --- Streamlit UI ---

st.set_page_config(page_title="문서 기반 Gemini 챗봇 (ChromaDB)", layout="centered")
st.title("문서 기반 Gemini 챗봇 📚 (ChromaDB)")

# 앱 시작 시 PDF 처리 실행
collection, loaded_files = load_and_process_pdfs_from_docs()

with st.sidebar:
    st.header("로드된 문서 목록")
    if loaded_files:
        for filename in loaded_files:
            st.info(filename)
    else:
        st.warning("로드된 문서가 없습니다.")

MODEL_NAME = "gemini-2.5-flash"

if "messages" not in st.session_state:
    initial_message = f"`docs` 폴더의 {len(loaded_files)}개 문서에 대한 내용이 준비되었습니다. 무엇이든 질문하세요."
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("문서 내용에 대해 질문하세요..."):
    st.empty() 
    # 사용자 메시지를 채팅 기록에 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
           #  message_placeholder = st.empty()
            
            relevant_chunks = get_relevant_chunks(prompt, collection)
            context = "\n\n".join(relevant_chunks)
            
            rag_prompt = f"""
            당신은 전산관리규정 문서 내용을 바탕으로 질문에 답변하는 AI 어시턴트입니다.
            반드시 아래 제공된 "문서 내용"을 근거로 해서 답변해야 합니다.
            문서 내용에 없는 정보는 답변하지 말고, "문서에서 관련 정보를 찾을 수 없습니다."라고 솔직하게 말하세요.        
            하지만, 문서 내용이 충분하지 않으면 일반적인 상식에 기반하여 답변할 수 있습니다.
            답변 후 답변과 관련된 문서 내용을 간략히 요약해서 "참고 문서"로 제공하세요.
            ---
            [문서 내용]
            {context}
            ---

            [질문]
            {prompt}
            """
            
            try:
                model = genai.GenerativeModel(MODEL_NAME)
                response = model.generate_content(rag_prompt)
                full_response = response.text
                
                # message_placeholder.markdown(full_response) <-- 이 부분을 삭제합니다.
                
                # 생성된 답변을 Streamlit Chat UI에 표시
                st.markdown(full_response) 
                
                # 답변 완료 후 session_state에 추가
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_message = f"Gemini API 호출 중 오류가 발생했습니다: {e}"
                # 오류 메시지를 Streamlit Chat UI에 표시
                st.error(error_message) 
                st.session_state.messages.append({"role": "assistant", "content": error_message})