import streamlit as st

# 사이드바
st.sidebar.title("챗봇")
st.sidebar.write("버전 1.0")

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
    # 사용자 메시지를 채팅 기록에 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 봇의 응답
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = f"봇 : {prompt}"
            st.markdown(response)

    # 봇의 응답을 채팅 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})