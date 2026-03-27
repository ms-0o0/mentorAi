import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# .env 파일에 저장된 API 키 로드
load_dotenv()

def create_rag_chain(pdf_path):

    # 1. 문서 로드
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # 2. 문서 분할 (개선)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # ↑ 늘림 (문맥 유지)
        chunk_overlap=150,
    )
    split_docs = text_splitter.split_documents(docs)

    # 3. 벡터 저장 (캐싱 가능 구조)
    embeddings = OpenAIEmbeddings()

    db_path = f"{pdf_path}.faiss"

    if os.path.exists(db_path):
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(db_path)

    # 4. Retriever 개선
    retriever = vectorstore.as_retriever(
        search_type="mmr",   # 다양성 고려 (중요)
        search_kwargs={"k": 5}
    )

    # 5. 프롬프트 개선 (핵심)
    template = """
        너는 문서를 기반으로 정확하게 답변하는 AI야.

        다음 규칙을 반드시 지켜:
        1. 주어진 context만 사용해
        2. 모르면 "문서에 없습니다"라고 말해
        3. 최대한 구체적으로 답해
        4. 근거도 함께 설명해

        [문서]
        {context}

        [질문]
        {question}

        [답변]
        """

    prompt = ChatPromptTemplate.from_template(template)

    # 6. LLM 개선
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.1   # 약간 창의성
    )

    # 7. 체인 구성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain