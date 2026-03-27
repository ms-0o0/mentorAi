import streamlit as st
import os
from rag_module import create_rag_chain

st.set_page_config(page_title="RAG 멘토링 챗봇", layout="wide")

st.title("🤖 PDF 기반 RAG 시스템")
st.markdown("업로드한 문서에 대해 질문해 보세요.")

# 1. 사이드바: 파일 업로드 및 가공
with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])

# 2. 파일 업로드 시 RAG 체인 초기화
if uploaded_file:
    temp_path = f"./temp/{uploaded_file.name}"

    # 폴더 없으면 생성
    os.makedirs("./temp", exist_ok=True)

    # 파일이 없을 때만 저장 (중복 방지)
    if not os.path.exists(temp_path):
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # RAG 체인 생성
    if "rag_chain" not in st.session_state:
        with st.spinner("문서 분석 중..."):
            st.session_state.rag_chain = create_rag_chain(temp_path)
        st.success("분석 완료!")

    # 3. 채팅 인터페이스 (메시지 이력 관리)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = st.session_state.rag_chain.invoke(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("왼쪽 사이드바에서 PDF 파일을 업로드하면 대화가 시작됩니다.")