# 1단계: 빌더(Builder) 스테이지
FROM python:3.10-slim AS builder

WORKDIR /app

# 빌드에 필요한 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 설치 준비
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .

# 종속성 설치 (의존성 파일들만 미리 생성)
# Pytorch는 용량이 크므로 명시적으로 CPU 버전 설치
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2단계: 런타임(Runtime) 스테이지
FROM python:3.10-slim

WORKDIR /app

# 런타임에 필요한 최소한의 시스템 패키지만 설치 (curl은 헬스체크용)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 빌더 스테이지에서 빌드된 파이썬 패키지(Wheel) 복사 및 설치
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    torch torchvision \
    && rm -rf /wheels

# 소스 코드 복사
COPY . .

# Streamlit 포트 노출
EXPOSE 8501

# 컨테이너 실행 시 Streamlit 실행
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]