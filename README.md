# Cloud AI Toy Project v2–v3

**(LangChain + pgvector RAG Agent)**

## 프로젝트 개요

이 프로젝트는 **Cloud AI 팀 지원용 토이 프로젝트**로,
AWS 환경으로 확장 가능한 **엔드투엔드 RAG + Agent 데모 서비스**를 구현하는 것이 목표다.

* **v2**: pgvector 기반 RAG 파이프라인
* **v3**: v2 위에 *판단·검증·재시도 로직을 갖춘 Agent 레이어* 추가

Docker · PostgreSQL(pgvector) · LangChain · Claude LLM을 사용해
**재현성, 안정성, 실무 적용 가능성**을 중점으로 설계되었다.

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
│       ├── main.py          # FastAPI 엔트리 (/health, /query, /agent_query)
│       ├── settings.py      # 환경변수 및 상수 관리
│       ├── vectorstore.py   # Embedding + PGVector 연결
│       ├── ingest_pg.py     # data/processed → pgvector 인덱싱
│       ├── rag_chain.py     # Retriever + Claude 응답 생성
│       └── agent_chain.py   # 판단·검증·재시도 로직을 포함한 RAG Agent
│
├── data/
│   ├── raw/                 # 원본 PDF
│   └── processed/           # 전처리된 JSON chunk
│
├── db/
│   └── init.sql             # 초기 테이블 생성 query
│
├── docker-compose.yml
├── .env.example
├── .env
└── README.md
```

---

## 데이터 흐름

### 1. PDF 전처리 (v1 결과 재사용)

* PDF → 텍스트 추출 → chunk 분할
* 결과를 `data/processed/*.json`으로 저장

---

### 2. 인덱싱 (`ingest_pg.py`)

* `data/processed/*.json` 로드
* LangChain `Document` 객체로 변환
* `PGVector.from_documents(..., pre_delete_collection=True)`
* 컬렉션 이름: `nds_food_safety_docs`

---

### 3. v2 RAG 질의 처리 (`/query`)

1. 사용자 질문 수신
2. pgvector 기반 k-NN 검색
3. 검색된 문서를 컨텍스트로 Claude 호출
4. 답변 + 사용된 컨텍스트 반환

> 단일 검색 + 단일 생성 구조

---

### 4. v3 Agent 질의 처리 (`/agent_query`)

v3에서는 v2 RAG 위에 **Agent 레이어**를 추가한다.

Agent는 다음을 수행한다:

1. **질문 의도 판단**

   * 요약 요청 여부 판단
2. **조건부 검색**

   * 1차 검색 실패 시 질의 재작성 후 재검색
3. **응답 생성**

   * 요약 강도, bullet 개수 제한 적용
4. **출력 검증**

   * bullet-only 구조 강제
   * 각 bullet에 `(doc_id, chunk_id)` 근거 필수
5. **재시도 루프**

   * 규칙 위반 시 최대 N회 재생성
6. **안전 종료**

   * 끝까지 실패 시 “문서 근거가 부족하다” 반환

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

---

#### v2 RAG Query

```bash
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"식품 제조 위생 점검 기준을 요약해줘","top_k":5}' \
| python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

---

#### v3 Agent Query (권장)

```bash
curl -s http://localhost:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{"query":"식품 제조 위생 점검 기준을 요약해줘","top_k":5}' \
| python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

Agent 응답에는 다음 정보가 포함된다:

* `decision`: Agent의 판단 결과

  * 요약 여부
  * 검색 라운드 수
* `answer`: 검증된 최종 답변
* `contexts`: 실제 사용된 문서 chunk

---

## 주요 설계 포인트

### 문서 기반 안전성

* 컨텍스트에 **명시적으로 존재하는 내용만** 사용
* 근거 없는 응답은 구조적으로 차단

### Agent 판단 로직

* 질문 의도에 따른 요약 강도 제어
* 검색 실패/출력 위반 시 자동 재시도

### 출력 검증

* 요약 모드에서는 **heading + bullet-only**
* 각 bullet에 `(doc_id, chunk_id)` 필수
* 검증 실패 시 재생성

### 완전 재현 가능 환경

* Docker + env 기반 실행
* 로컬 / CI / 타 환경 동일 동작 보장

---

## 향후 확장 아이디어

* 문서 분류 모델(TF-IDF / BERT) 기반 검색 필터링
* 컨텍스트 재랭킹(Reranker)
* 세션 단위 메모리(대화형 Agent)
* AWS 아키텍처(ECS, RDS, Bedrock) 매핑 문서화

---

## 현재 상태

* v2 RAG 파이프라인 구현 완료
* v3 판단형 RAG Agent 구현 완료
* 로컬 Docker 환경에서 정상 동작 확인
* **Agent 수준의 판단·검증·재시도 구조 확보**

---
