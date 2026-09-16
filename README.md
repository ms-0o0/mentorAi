# 🤖 RAG Mentor AI - PDF 기반 맞춤형 학습 & 연구 지원 챗봇

<p align="center">
  <img src="https://img.shields.io/badge/RAG_Mentor_AI-v1.0.0-4A90E2?style=for-the-badge&logo=robot-framework&logoColor=white" alt="Project Banner" />
</p>

> **RAG Mentor AI**는 업로드된 PDF 문서를 바탕으로 사용자 역할(학생/교수)과 목적(질문/요약/문제 생성)에 맞춰 지능형 답변을 제공하는 **로컬 프라이버시 중심의 Retrieval-Augmented Generation (RAG) 챗봇 시스템**입니다.

---

## 📌 목차
- [기술 스택 (Tech Stack)](#-기술-스택-tech-stack)
- [주요 기능 (Key Features)](#-주요-기능-key-features)
- [기술 선택 사유 (Technical Decision & Rationale)](#-기술-선택-사유-technical-decision--rationale)
- [시스템 아키텍처 (System Architecture)](#-시스템-아키텍처-system-architecture)
- [실행 화면 (Screenshots)](#-실행-화면-screenshots)
- [프로젝트 구조 (Directory Structure)](#-프로젝트-구조-directory-structure)
- [실행 방법 (Getting Started)](#-실행-방법-getting-started)

---

## 🛠 기술 스택 (Tech Stack)

### Language & Framework
![Python](https://img.shields.io/badge/Python_3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

### Vector Database & Embedding
![FAISS](https://img.shields.io/badge/FAISS-00599C?style=for-the-badge&logo=meta&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch_(CPU)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

### LLM & Infra
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![Qwen 2.5](https://img.shields.io/badge/Qwen_2.5_7B-61DAFB?style=for-the-badge&logo=alibabacloud&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## ✨ 주요 기능 (Key Features)

| 기능 | 설명 |
| :--- | :--- |
| **👤 Dual-Role Persona** | **학생 모드** (친절한 1타 강사 tone, 쉬운 비유와 단계적 설명)와 **교수 모드** (학술적 심층 분석 tone, 연구/강의 시사점) 간 즉시 전환 가능 |
| **🎯 Multi-Mode Operations** | **질문(Q&A)**, **문서 요약(Summary)**, **맞춤형 문제 생성(Quiz)** 3가지 실행 모드 제공 |
| **📌 출처(Page Citation) 추적** | 답변 생성에 참조된 PDF 문서의 정확한 페이지 정보(`X페이지`)를 자동 태깅하여 제공 |
| **⚡ 스마트 백엔드 캐싱** | PDF 문서 파일의 MD5 해시값을 기반으로 벡터 DB를 로컬 저장소에 인덱싱하여, 동일 문서 재업로드 시 즉시 대화 가능 |
| **🔒 100% 로컬 보안 환경** | 외부 Cloud API 전송 없이 로컬 LLM(Ollama) 및 로컬 Vector DB(FAISS)를 통해 데이터 유출 방지 |

---

## 💡 기술 선택 사유 (Technical Decision & Rationale)

본 프로젝트는 **독립적인 로컬 실행 환경**, **빠른 임베딩 검색**, **보안성** 및 **운용 효율성**을 최우선으로 고려하여 기술 스택을 구성하였습니다.

### 1. 🗄️ 백엔드 Vector DB: FAISS (Facebook AI Similarity Search)
> **"왜 외부 벡터 DB(Pinecone, Milvus 등)가 아닌 FAISS를 선택했는가?"**

- **초경량 및 인메모리 검색 최적화**: 
  단일 애플리케이션 및 로컬 RAG 환경에서는 네트워크 오버헤드가 있는 클라우드 DB보다 **메모리 단에서 즉각적인 벡터 연산을 수행하는 FAISS**가 훨씬 뛰어난 검색 속도를 자랑합니다.
- **로컬 디스크 영속화 & MD5 해시 캐싱**:
  - PDF 업로드 시 `hashlib.md5(pdf_path)`를 통해 고유 해시값을 생성하고 `./temp/{hash}.faiss` 파일로 저장합니다.
  - 동일한 문서가 다시 업로드될 경우, 대용량 문서의 청킹 및 임베딩 재연산 과정 없이 **로컬 FAISS 인덱스를 즉시 로드(load_local)**하여 응답 지연 시간을 최소화했습니다.
- **MMR(Maximal Marginal Relevance) 검색 지원**:
  - 단순 코사인 유사도 검색 시 발생할 수 있는 중복 정보 검색 문제를 방지하기 위해 `search_type="mmr"`, `k=3`, `fetch_k=10` 설정을 적용하여 **관련성과 다양성을 모두 확보**했습니다.

### 2. 🦙 로컬 LLM 엔진: Ollama + Qwen 2.5 (7B)
- **데이터 보안 및 프라이버시**:
  - 대학 강의 자료, 연구 논문 등 외부 유출이 민감한 PDF 문서를 다루기 때문에, OpenAI 등 외산 Cloud API 대신 **100% On-Premise/Local LLM**인 Ollama를 선택했습니다.
- **한국어 지시 이행 능력 (Instruction Following)**:
  - Qwen 2.5 7B 오픈소스 모델은 7B 파라미터 체급 대비 한국어 문장 표현력이 뛰어나며, 프롬프트에서 지정한 **Persona(학생/교수) 및 Markdown 출력 구조(객관식/단답형 퀴즈 양식)**를 왜곡 없이 엄격하게 준수합니다.

### 3. 🔤 한국어 특화 임베딩: `jhgan/ko-sroberta-multitask`
- **한국어 문맥 파악 SOTA**:
  - 범용 영어 임베딩 모델과 달리 한국어 문장 간 유의미한 유사도를 정확히 계산하는 다국어/한국어 특화 SROBERTA 모델을 적용했습니다.
- **PyTorch CPU 호환성**:
  - Docker 컨테이너 및 일반 CPU 환경에서도 별도의 GPU 장비 없이 신속한 벡터화 연산이 가능합니다.

### 4. 🎨 프론트엔드 UI: Streamlit
- **세션 기반 대화 이력 관리**:
  - `st.session_state`를 활용해 대화 히스토리 유지, 초기화 버튼 기능, 역할/모드 변경 시 체인 재구성을 손쉽게 구현할 수 있습니다.

---

## 🏗 시스템 아키텍처 (System Architecture)

```yaml
                            [ PDF 파일 업로드 ]
                                     │
                                     ▼
                      [ PyMuPDF 로드 및 텍스트 정제 ]
                           (정규식 clean_text 적용)
                                     │
                                     ▼
                    [ RecursiveTextSplitter 청킹 ]
                        (Chunk: 400, Overlap: 100)
                                     │
                                     ▼
                   [ PDF 파일 MD5 해시 생성 (Hashing) ]
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
         (로컬 DB 존재: Cache Hit)         (새 문서: Cache Miss)
                    │                                 │
                    ▼                                 ▼
          [ FAISS.load_local ]              [ HuggingFace 임베딩 생성 ]
                    │                                 │
                    │                                 ▼
                    │                       [ FAISS.save_local 저장 ]
                    └────────────────┬────────────────┘
                                     ▼
                          [ FAISS Vector Store ]
                                     │
                                     ▼
                        [ MMR Retriever (k=3) ]
                                     │
                                     ▼
                         [ Dynamic Prompt Router ]
                     (학생/교수 Persona & 모드 분기)
                                     │
                                     ▼
                        [ Ollama (Qwen 2.5 7B) ]
                                     │
                                     ▼
                    [ Streamlit 대화형 UI 응답 출력 ]
                        (+ 페이지 번호 출처 표시)
```

---

## 📸 실행 화면 (Screenshots)
<details open>
<summary><b>📷 실행 화면 갤러리 (클릭하여 펴기/접기)</b></summary>

<br/>

| 화면 구분 | 캡처 이미지 (Placeholder) | 주요 기능 설명 |
| :---: | :--- | :--- |
| **여러 모드 선택 창** | `![모드 선택 창](docs/screenshots/모드%20선택.png)` | 모드 선택 및 PDF 자료 업로드 창 |
| **문서 요약** | `![요약](docs/screenshots/요약.png)` | 문서 전체 요약  |
| **질문 답변** | `![질문 답변](docs/screenshots/질문.png)` | 학생의 질문에 따라 질문을 답을 함 |

</details>

---

## 📁 프로젝트 구조 (Directory Structure)

```text
mentorAi
├── 📄 app.py                  # Streamlit 메인 UI 및 세션/이력 관리
├── 📄 rag_module.py           # RAG 체인, FAISS 인덱싱, 동적 프로필/모드 라우터
├── 📄 requirements.txt        # 프로젝트 파이썬 의존성 패키지
├── 🐳 Dockerfile              # Multi-stage Docker 빌드 정의
├── 🐙 docker-compose.yml      # Ollama 서비스 및 RAG 챗봇 멀티 컨테이너 오케스트레이션
├── 🔐 .env                    # 환경 변수 설정
├── 📁 temp/                   # 업로드된 PDF 파일 및 FAISS 벡터 DB 캐시 저장소
└── 📁 docs/
    └── 📁 screenshots/        # README용 실행 화면 캡처 이미지 저장 폴더
```

---

## 🚀 실행 방법 (Getting Started)

### 1. 사전 준비 사항 (Prerequisites)
- Docker 및 Docker Desktop 설치
- 또는 Python 3.10+ 환경 (로컬 직접 실행 시)

---

### 🐳 Option A: Docker Compose를 이용한 실행 (추천)

Ollama 컨테이너와 RAG 챗봇 시스템을 한 번에 구동합니다.

1. **저장소 클론 및 이동**
   ```bash
   git clone https://github.com/your-username/mentorAi.git
   cd mentorAi
   ```

2. **Docker Compose 실행**
   ```bash
   docker-compose up --build
   ```
   > 💡 **최초 구동 시:** `init-ollama` 컨테이너가 `Qwen 2.5 7B` 모델을 자동으로 다운로드하므로 몇 분 정도 시간이 소요될 수 있습니다.

3. **웹 브라우저 접속**
   - Streamlit UI: `http://localhost:8501`
   - Ollama API Server: `http://localhost:11434`

---

### 🐍 Option B: 로컬 Python 환경 직접 실행

1. **의존성 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama 로컬 서버 구동 및 모델 다운로드**
   ```bash
   ollama pull qwen2.5:7b
   ```

3. **Streamlit 애플리케이션 실행**
   ```bash
   streamlit run app.py
   ```

---

## 📝 라이선스 & 문의 (License & Contact)
- **License**: MIT License
- **Author**: 김민수 (Minsu Kim)
