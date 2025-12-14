# Cloud AI Toy Project v2 (LangChain + pgvector RAG)

## 프로젝트 개요

이 프로젝트는 **농심 NDS Cloud AI 팀 지원용 토이 프로젝트**로,
AWS 환경으로 확장 가능한 **엔드투엔드 RAG(Retrieval-Augmented Generation) 데모 서비스**를 구현하는 것이 목표다.

v2는 기존 v1(FAISS 기반, LangChain 미사용)과 완전히 분리된 **새로운 구현 라인**이며,
Docker · PostgreSQL(pgvector) · LangChain · Claude LLM을 사용해
실무에 가까운 구조와 재현성을 중점으로 설계되었다.

---

## 기술 스택

* **Language**: Python 3.12
* **API Server**: FastAPI
* **LLM Orchestration**: LangChain
* **LLM**: Anthropic Claude (ChatAnthropic)
* **Embedding Model**: `intfloat/multilingual-e5-base`
* **Vector Store**: PostgreSQL + pgvector (`pgvector/pgvector:pg16`)
* **Infra / Runtime**: Docker, docker-compose
* **Environment Management**: `.env`, `.env.example`, `.gitignore`

---

## 디렉터리 구조

```text
cloud-ai-toy/
├── app_v2/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app_v2/
│       ├── __init__.py
│       ├── main.py          # FastAPI 엔트리 (/health, /query)
│       ├── settings.py      # 환경변수 및 상수 관리
│       ├── vectorstore.py   # Embedding + PGVector 연결
│       ├── ingest_pg.py     # data/processed → pgvector 인덱싱
│       └── rag_chain.py     # Retriever + Claude 응답 생성
│
├── data/
│   ├── raw/                 # 원본 PDF
│   └── processed/           # 전처리된 JSON chunk
│
├── db/
│   └── init.sql             # CREATE EXTENSION vector
│
├── docker-compose.yml
├── .env.example
├── .env                     # gitignore 대상
└── README.md
```

---

## 데이터 흐름 (RAG 파이프라인)

1. **PDF 전처리 (v1 결과 재사용)**

   * PDF → 텍스트 추출 → chunk 분할
   * 결과를 `data/processed/*.json`으로 저장

2. **인덱싱 (`ingest_pg.py`)**

   * `data/processed/*.json` 로드
   * LangChain `Document` 객체로 변환
   * `PGVector.from_documents(..., pre_delete_collection=True)`
   * 컬렉션 이름: `nds_food_safety_docs`

3. **질의 처리 (`/query`)**

   * 사용자 질문 수신
   * pgvector 기반 k-NN 검색
   * 검색된 문서를 컨텍스트로 Claude 호출
   * 답변 + 사용된 컨텍스트 반환

---

## 실행 방법

### 1. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 예시:

```env
POSTGRES_DB=ragdb
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpassword

DATABASE_URL=postgresql+psycopg://raguser:ragpassword@db:5432/ragdb
ANTHROPIC_API_KEY=YOUR_API_KEY
```

---

### 2. 컨테이너 실행

```bash
docker compose up -d --build
```

상태 확인:

```bash
docker compose ps
```

---

### 3. 인덱싱 실행

```bash
docker compose run --rm app python3 -m app_v2.ingest_pg
```

성공 시:

```
OK: indexed N documents into collection='nds_food_safety_docs'
```

---

### 4. API 테스트

#### Health Check

```bash
curl http://localhost:8000/health
```

#### RAG Query

```bash
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"식품 제조 위생 점검 기준을 요약해줘","top_k":5}' \
| python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

---

## 주요 설계 포인트

* **완전 재현 가능 환경**

  * Docker + env 기반 실행
  * 로컬/CI/타 환경에서 동일 동작 보장

* **LangChain 공식 컴포넌트 사용**

  * PGVector, Retriever, ChatAnthropic

* **추측 없는 RAG**

  * 문서 컨텍스트 기반 응답
  * 컨텍스트가 없을 경우 근거 부족 명시

* **확장 고려 설계**

  * 분류 모델 기반 검색 필터링 가능
  * AWS (ECS, OpenSearch, Bedrock)로 자연스럽게 이전 가능

---

## 향후 확장 아이디어

* 문서 분류 모델(TF-IDF / BERT) 연계 검색
* 답변에 출처(doc_id, chunk_id) 강제
* 컨텍스트 길이 제한 및 재랭킹
* AWS 아키텍처 매핑 문서화

---

## 상태

* v2 기본 RAG 파이프라인 구현 완료
* 로컬 Docker 환경에서 정상 동작 확인
