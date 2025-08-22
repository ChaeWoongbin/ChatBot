

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

def generate_answer(query):
    start = time.time()
    msgManager.append_msg(query)
    
    # 스트리밍으로 답변 생성
    print("답변: ", end="", flush=True)
    full_answer = ""

    is_first_chunk = True
    start_answer_time = None

    msg = list(msgManager.queue)

    for response in ollama.chat(model=model, messages=msg, stream=True):
        if is_first_chunk:
            start_answer_time = time.time()
            is_first_chunk = False
            
        chunk = response["message"]["content"]
        print(chunk, end="", flush=True)
        full_answer += chunk
    
    print()
    
    # LLM 추론 시간과 답변 시작 시간 출력
    print(f"LLM 전체 추론: {time.time() - start:.2f}초")
    if start_answer_time:
        print(f"답변 시작 시간: {start_answer_time - start:.2f}초")
        
    return full_answer

def chat_loop():

    print("챗봇 시작! 질문 입력 (종료하려면 'exit' 입력):")

    while True:
        query = input("> ")
        if query.lower() == "exit":
            print("챗봇 종료!")
            break

        start_time = time.time()

        generate_answer(query)

        print(f"[총 소요 시간: {time.time() - start_time:.2f}초]\n")


# 실행
if __name__ == "__main__":
    chat_loop()