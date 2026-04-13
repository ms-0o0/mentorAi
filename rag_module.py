import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

MODE_QA = "질문"
MODE_QUIZ = "문제 생성"
MODE_SUMMARY = "요약"

# # .env 파일에 저장된 API 키 로드
# load_dotenv()

def clean_text(text):
    # 이상한 문자 제거
    text = re.sub(r'[^\w\s가-힣.,!?()\-\n]', '', text)
    return text

# 7. 체인 구성
def format_docs(docs):
    context = ""
    sources = []
    seen = set()

    for d in docs:
        cleaned = clean_text(d.page_content)

        # 🔥 중복 제거 핵심
        key = cleaned[:150]
        if key in seen:
            continue
        seen.add(key)

        context += cleaned + "\n\n"

        page = d.metadata.get("page", "unknown")
        sources.append(f"{page}페이지")

    return context, sorted(set(sources), key=lambda x: int(x.replace("페이지", "")) if x != "unknown" else 999)

def create_rag_chain_internal(retriever, llm, role="student"):
    
    template_student = """
        # 역할
        너는 학생의 학습을 돕는 친절한 AI 1타 강사다.

        # 제약조건
        1. 반드시 제공된 문서(context)만 기반으로 답변한다.
        2. 쉬운 비유와 친절한 말투(~해요, ~습니다)를 사용한다.
        3. 단계적으로 차근차근 설명한다.
        4. 반드시 한국어로만 답변한다.
        5. 같은 문장 반복 금지
        6. 동일한 내용 여러 번 설명 금지
        7. 중복 내용은 제거하고 요약할 것

        # 답변 형식
        ✨ [핵심 요약]
        - 질문에 대한 가장 핵심적인 답변

        📖 [상세하고 쉬운 설명]
        - 개념과 과정을 차근차근 풀어서 설명

        💡 [예시/비유]
        - 내용 이해를 돕는 일상적인 비유나 예시

        # 문서
        {context}

        # 질문
        {question}

        # 답변
    """
    
    template_professor = """
        # 역할
        너는 교수의 강의 준비 및 연구를 지원하는 전문 학술 AI 조교다.

        # 제약조건
        1. 반드시 문서(context) 기반으로 사실에 입각하여 답변한다.
        2. 깊이 있는 전문 용어를 활용하고, 학술적인 문체(~이다, ~함)를 사용한다.
        3. 논리적으로 구조화된 답변을 제공한다.
        4. 반드시 한국어로만 답변한다.
        5. 같은 문장 반복 금지
        6. 동일한 내용 여러 번 설명 금지
        7. 중복 내용은 제거하고 요약할 것

        # 답변 형식
        📌 [논점 핵심 요약]
        - 질문에 대한 학술적이고 간결한 핵심 정리

        🔎 [심층 분석]
        - 문서를 바탕으로 한 개념의 논리적, 구조적 상세 설명

        🎓 [강의/연구 시사점]
        - 이 개념을 강의나 연구에 적용할 때 강조해야 할 포인트

        # 문서
        {context}

        # 질문
        {question}

        # 답변
    """

    if role == "student":
        template = template_student
    elif role == "professor":
        template = template_professor
    else:
        template = template_student  # 기본값

    # # 1. 문서 로드
    # loader = PyMuPDFLoader(pdf_path)
    # docs = loader.load()

    # # 2. 문서 분할 (개선)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=700,
    #     chunk_overlap=200,
    #     separators=["\n\n", "\n", ".", " "]
    # )
    # split_docs = text_splitter.split_documents(docs)

    # # 3. 벡터 저장 (캐싱 가능 구조)
    # # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(
    #     # model_name="sentence-transformers/all-MiniLM-L6-v2"
    #     model_name="jhgan/ko-sroberta-multitask"
    # )

    # import hashlib
    # safe_filename = hashlib.md5(pdf_path.encode('utf-8')).hexdigest()
    # db_path = f"./temp/{safe_filename}.faiss"

    # if os.path.exists(db_path):
    #     vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    # else:
    #     vectorstore = FAISS.from_documents(split_docs, embeddings)
    #     vectorstore.save_local(db_path)

    # # 4. Retriever 개선
    # retriever = vectorstore.as_retriever(
    #     search_type="mmr",   # 다양성 고려 (중요)
    #     search_kwargs={"k": 5}
    # )

    prompt = ChatPromptTemplate.from_template(template)

    # 6. LLM 개선
    # 기존 코드 주석 처리 (나중에 유료 버전 쓸 때 주석을 해제하세요)
    # llm = ChatOpenAI(
    #     model_name="gpt-4o",
    #     temperature=0.1   # 약간 창의성
    # )

    # 무료 버전 LLM (Google Gemini) 주석 처리
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",   # 🔥 이걸로 변경
    #     temperature=0.1
    # )

    # 로컬 모델 (Ollama) 사용

    # def get_llm():
    #     return ChatOllama(
    #         model="qwen2.5",
    #         temperature=0.1
    #     )

    def rag_chain_with_context(question, context):
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })
        return response


    return rag_chain_with_context


def route(question, mode, retriever, rag_chain_with_context, quiz_chain, summary_chain):
    if mode == MODE_QUIZ:
         # 1. 문서 가져오기
        docs = retriever.invoke("핵심 개념 요약")
        context, _ = format_docs(docs)

        # 2. 먼저 요약 (🔥 핵심)
        summary = summary_chain.invoke({"context": context})

        # 3. 요약 기반으로 문제 생성
        return quiz_chain.invoke({
            "context": summary,
            "question": question
        })

    elif mode == MODE_SUMMARY:
        docs = retriever.invoke("전체 요약")
        context, _ = format_docs(docs)
        return summary_chain.invoke({"context": context})

    else:
        docs = retriever.invoke(question)
        context, sources = format_docs(docs)
        response = rag_chain_with_context(question, context)
        return response + "\n\n[출처]\n" + ", ".join(sources)

def create_quiz_chain(llm, role="student"):
    if role == "professor":
        template = """
        # 역할
        너는 대학교수의 강의 자료 준비 및 시험 출제를 지원하는 전문 AI 조교다.

        # 명령문
        주어진 문서를 바탕으로, 학부생 또는 대학원생의 전문 지식을 평가하기 위한 난이도 높고 변별력 있는 문제를 생성해라.

        # 제약조건
        1. 반드시 문서(context) 내용만 기반으로 문제를 만든다.
        2. 문제의 개수는 반드시 최대 3개까지 낸다.
        3. 단순 개념 암기보다는 응용 및 깊은 이해를 묻는 문제를 포함한다.
        4. 객관식은 보기 4개, 매력적인 오답을 포함하여 출제한다.
        5. 중간/기말고사에 활용 가능하도록 문제, 정답, 그리고 채점 기준(상세 해설)을 명확히 제시한다.
        6. 반드시 한국어로만 답변한다.
        7. 같은 문장 반복 금지
        7. 동일한 내용 여러 번 설명 금지
        8. 중복 내용은 제거하고 요약할 것

        # 출력형식
        ## 📝 객관식 문제
        ### 1. [문제]
        a) [보기]
        b) [보기]
        c) [보기]
        d) [보기]
        **정답:** a
        **해설 및 출제 의도:** [해설]

        ---

        ## ✍️ 서술형 문제
        ### 2. [문제]
        **모범 답안:** [답안]
        **채점 기준(핵심 키워드):** [키워드 및 해설]

        # 문서
        {context}
        """
    else:

          template = """
            # 역할
            너는 수학 문제를 정확하게 출제하는 교사다.

            # 사용자 요구
            {question}

            # 절대 규칙 (어기면 실패)
            1. 문제는 정확히 2개만 생성한다
            2. 반드시 "사용자가 요구한 연산"만 사용한다
            3. 문제는 반드시 "풀어야 하는 질문 형태"로 작성한다
            (❌ 식만 던지기 금지)
            (⭕ ~을 계산하시오 / 전개하시오)
            4. 객관식은 "문제 1개 + 보기 4개 + 정답 1개" 구조만 허용
            5. 보기에는 정답이 아닌 오답 3개 포함
            6. 단답형도 반드시 계산 문제로 만든다
            7. 추가 문제 생성 금지
            8. 출력 형식 외 텍스트 절대 금지

            # 문제 생성 규칙
            - 1번: 객관식 (전개 or 계산 문제)
            - 2번: 단답형 (전개 or 계산 문제)

            # 출력 형식 (절대 수정 금지)
            ## 📌 객관식 퀴즈
            ### 1. 문제
            (다항식 곱셈 문제를 한 줄로 제시)

            a) (정답 포함 4지선다)
            b) 
            c) 
            d) 

            정답: (a/b/c/d 중 하나)
            해설: (간단한 계산 과정)

            ---

            ## 💡 단답형 퀴즈
            ### 2. 문제
            (다항식 곱셈 문제)

            정답: (계산 결과)
            해설: (계산 과정)

            # 문서
            {context}
            """

    prompt = ChatPromptTemplate.from_template(template)

    return prompt | llm | StrOutputParser()

def create_summary_chain(llm, role="student"):
    if role == "professor":
        template = """
        # 역할
        너는 전문 문헌을 분석하여 교수의 강의 연구를 돕는 AI 학술 조교다.

        # 명령문
        주어진 문서를 학술적이고 논리적인 구조로 요약하라.

        # 제약조건
        1. 문서(context) 기반으로 작성하며, 핵심 논점 및 발견을 정확히 짚어낸다.
        2. 학술적인 문체(~이다, ~함)를 사용한다.
        3. 강의 자료 목차나 연구 요약본으로 바로 쓸 수 있도록 체계적으로 구조화한다.
        4. 반드시 한국어로만 답변한다.
        5. 같은 문장 반복 금지
        6. 동일한 내용 여러 번 설명 금지
        7. 중복 내용은 제거하고 요약할 것   

        # 출력형식
        📚 **[논점 요약]**
        - 문서의 전체적인 핵심 주장을 한 문단으로 요약

        🎯 **[주요 발견 및 핵심 개념]**
        - (핵심 개념): (학술적 정의 및 중요성)

        📊 **[연구/강의 시사점]**
        - 문서 내용을 기반으로 한 학술적 시사점

        # 문서
        {context}
        """
    else:
        template = """
        # 역할
        너는 복잡한 내용을 아주 쉽게 풀어서 설명해주는 1타 강사 AI 튜터다.

        # 명령문
        주어진 문서를 학생들이 직관적으로 이해할 수 있도록 쉽고 명확하게 요약하라.

        # 제약조건
        1. 문서(context) 기반으로 핵심만 추려서 작성한다.
        2. 중학생~고등학생도 이해할 수 있을 만큼 쉽고 친근한 말투(~해요, ~습니다)를 사용한다.
        3. 비유나 예시를 들 수 있으면 활용하여 이해를 일상적으로 돕는다.
        4. 반드시 한국어로만 답변한다.
        5. 같은 문장 반복 금지
        6. 동일한 내용 여러 번 설명 금지
        7. 중복 내용은 제거하고 요약할 것

        # 출력형식
        ✨ **[한 줄 핵심 파악]**
        - 전체 내용을 가장 쉽게 한 문장으로 요약

        🧠 **[꼭 알아야 할 내용]**
        - (핵심 개념): (아주 쉬운 설명)

        💡 **[쉬운 마무리 결론]**
        - 결론과 기억해야 할 포인트 요약

        # 문서
        {context}
        """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

def create_router_chain(retriever, llm, role, all_docs):

    rag_chain_with_context = create_rag_chain_internal(retriever, llm, role)
    quiz_chain = create_quiz_chain(llm, role)
    summary_chain = create_summary_chain(llm, role)

    def router(question, mode):
        return route(
            question,
            mode,
            retriever,
            rag_chain_with_context,
            quiz_chain,
            summary_chain
        )

    return router

def create_rag_chain(pdf_path, role="student"):

    # 1. 문서 로드
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # 2. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = text_splitter.split_documents(docs)

    # 3. 임베딩
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask"
    )

    import hashlib
    safe_filename = hashlib.md5(pdf_path.encode('utf-8')).hexdigest()
    os.makedirs("./temp", exist_ok=True)
    db_path = f"./temp/{safe_filename}.faiss"

    if os.path.exists(db_path):
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(db_path)

    # 4. retriever
    # retriever = vectorstore.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 3}
    # )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )

    # 5. LLM
    try:
        # ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        ollama_base_url = "http://localhost:11434"
        llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0.1,
            base_url=ollama_base_url,
            timeout=180
        )
    except Exception as e:
        raise Exception(f"Ollama 서버 응답 없음: {str(e)}\n\n*해결 가이드*\n1. Docker 컨테이너가 정상 구동 중인지 확인하세요.\n2. 초기 구동 시 모델 다운로드에 시간이 걸릴 수 있습니다.")

    # 6. Router
    router_chain = create_router_chain(retriever, llm, role, docs)

    return router_chain
