import streamlit as st
import ollama
import time
from collections import deque



# 메세지 매니저 클래스 구현
class Message_manager:
    def __init__(self):
        self._system_msg = {"role": "system", "content": ""}
        self.queue = deque(maxlen=10)  # 최대 10개 대화 저장

    def create_msg(self, role, content):
        return {"role": role, "content": content}

    def system_msg(self, content):
        self._system_msg = self.create_msg("system", content)

    def append_msg(self, content):
        msg = self.create_msg("user", content)
        self.queue.append(msg)

msgManager = Message_manager()


# 시스템 메세지 등록
msgManager.system_msg(
    "당신은 유능한 챗봇입니다. 사용자의 질문에 대해 친절하고 정확하게 답변해주세요."
)

model = "exaone3.5"


# 사이드바
st.sidebar.title("챗봇")
st.sidebar.write("버전 1.1")

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
            msgManager.append_msg(prompt)
            
        
            # 스트리밍으로 답변 생성
            placeholder = st.empty()
            full_answer = f"봇 :"

            is_first_chunk = True
            start_answer_time = None

            msg = list(msgManager.queue)

            for response in ollama.chat(model=model, messages=msg, stream=True):
                if is_first_chunk:
                    start_answer_time = time.time()
                    is_first_chunk = False
                    
                chunk = response["message"]["content"]
                full_answer += chunk
                placeholder.markdown(full_answer)
            
            # LLM 추론 시간과 답변 시작 시간 출력
            st.success(f"LLM 전체 추론: {time.time() - start_time:.2f}초")
            if start_answer_time:
                st.success(f"답변 시작 시간: {start_answer_time - start_time:.2f}초")

            st.success(f"[총 소요 시간: {time.time() - start_time:.2f}초]\n")

    # 봇의 응답을 채팅 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
