import streamlit as st
import os
from rag_module import create_rag_chain

MODE_QA = "질문"
MODE_QUIZ = "문제 생성"
MODE_SUMMARY = "요약"

st.set_page_config(page_title="RAG 멘토링 챗봇", layout="wide")

st.title("🤖 PDF 기반 RAG 시스템")
st.markdown("업로드한 문서에 대해 질문해 보세요.")

# 1. 사이드바: 파일 업로드 및 가공
with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])

    role_display = st.selectbox(
        "👤 사용자 유형 선택",
        ["학생 (친절하고 쉬운 설명)", "교수 (전문적이고 심층적인 분석)"]
    )
    role = "student" if "학생" in role_display else "professor"

    mode = st.radio(
        "🎯 모드 선택",
        ["질문", "요약", "문제 생성"]
    )

# 2. 파일 업로드 시 RAG 체인 초기화
if uploaded_file:
    temp_path = f"./temp/{uploaded_file.name}"

    # 폴더 없으면 생성
    os.makedirs("./temp", exist_ok=True)

    # 파일이 없을 때만 저장 (중복 방지)
    if not os.path.exists(temp_path):
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # 3. 채팅 인터페이스 (메시지 이력 관리)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # RAG 체인 생성
    if (
        "rag_chain" not in st.session_state
        or st.session_state.get("role") != role
        or st.session_state.get("file_name") != uploaded_file.name
    ):
        with st.spinner("문서 분석 중..."):
            st.session_state.rag_chain = create_rag_chain(temp_path, role)
            st.session_state.role = role
            st.session_state.file_name = uploaded_file.name
        st.success("분석 완료!")
    

    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.rerun()

    # 기존 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요 (예: 개념 설명 / 요약 / 문제 생성)"):
        # 1. 사용자 질문 저장 및 즉시 출력
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 어시스턴트 답변 즉시 생성 및 출력
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    if mode == MODE_SUMMARY:
                        response = st.session_state.rag_chain(prompt, MODE_SUMMARY)
                    elif mode == MODE_QUIZ:
                        response = st.session_state.rag_chain(prompt, MODE_QUIZ)
                    else:
                        response = st.session_state.rag_chain(prompt, MODE_QA)
                except Exception as e:
                    response = f"오류 발생: {str(e)}"

                st.markdown(response)

        # 3. 어시스턴트 답변 세션 저장 (rerun 없이 자연스럽게 이어나감)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("왼쪽 사이드바에서 PDF 파일을 업로드하면 대화가 시작됩니다.")