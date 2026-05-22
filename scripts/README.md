# Scripts

주요 컨퍼런스 논문을 자동으로 수집하고 README.md를 갱신하는 스크립트 모음입니다.
데이터 출처: **DBLP** (논문 목록) + **Semantic Scholar** (초록 요약)

---

## 파일 구조

```
scripts/
├── config.py          # 검색 설정 (키워드, 컨퍼런스, 연도 범위)
├── crawler.py         # DBLP/Semantic Scholar 크롤러
├── readme_updater.py  # README.md 자동 생성기
└── main.py            # 진입점 (CLI)
```

---

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. API 키 설정 (선택)

API 키 없이도 동작하지만 **10배 느립니다** (1 req/sec → 10 req/sec).

```bash
cp .env.example .env
# .env 파일을 열어 키 입력
# SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

무료 키 발급: https://www.semanticscholar.org/product/api

### 3. 실행

```bash
# 기본 실행: 캐시 없는 데이터 수집 + README 갱신
python scripts/main.py

# README만 갱신 (수집 없이)
python scripts/main.py --readme-only
```

---

## 토픽 변경 방법

`scripts/config.py` 상단의 두 값만 수정하면 됩니다.

```python
# 논문 제목에서 찾을 키워드 (소문자, 부분 일치)
KEYWORDS = ['recommendation', 'recommender', 'recommend']

# README 제목·요약에 표시할 짧은 레이블
TOPIC_LABEL = 'Recommendation'
```

**예시 — LLM 논문 추적:**

```python
KEYWORDS    = ['llm', 'large language model', 'foundation model']
TOPIC_LABEL = 'LLM'
```

변경 후 캐시를 초기화하고 재수집합니다.

```bash
rm data/cache.json        # Windows: del data\cache.json
python scripts/main.py
```

---

## 컨퍼런스/연도 범위 변경

`scripts/config.py`에서 수정합니다.

```python
START_YEAR = 2022   # 수집 시작 연도
END_YEAR   = 2026   # 수집 종료 연도 (기본값: 현재 연도)
```

특정 컨퍼런스나 연도를 추가하려면 `CONFERENCES` 딕셔너리에 항목을 추가합니다.

```python
CONFERENCES = {
    'RecSys': {
        'display_name': 'RecSys',
        'dblp_id': 'recsys',          # DBLP conf 경로: conf/recsys/
        'full_name': 'ACM Conference on Recommender Systems',
    },
    # 새 컨퍼런스 추가 예시
    'ECIR': {
        'display_name': 'ECIR',
        'dblp_id': 'ecir',
        'full_name': 'European Conference on Information Retrieval',
    },
    ...
}
```

> `dblp_id`는 `https://dblp.org/db/conf/{dblp_id}/` URL에서 확인할 수 있습니다.
> 파일명 패턴이 다른 경우 `dblp_toc_prefix` 필드를 추가합니다 (예: NeurIPS).

---

## CLI 옵션

```
python scripts/main.py [옵션]

옵션:
  (없음)                    캐시 없는 항목만 수집 후 README 갱신
  --readme-only             수집 없이 README만 갱신
  --conf RecSys SIGIR       특정 컨퍼런스만 수집
  --year 2024 2025          특정 연도만 수집
  --refresh SIGIR 2024      특정 컨퍼런스/연도 강제 재수집 (캐시 무시)
```

**예시:**

```bash
# SIGIR 2024만 강제 재수집
python scripts/main.py --refresh SIGIR 2024

# KDD, SIGIR의 2023~2024년만 수집
python scripts/main.py --conf KDD SIGIR --year 2023 2024

# 캐시 데이터로 README만 새로 생성
python scripts/main.py --readme-only
```

---

## 데이터 흐름

```
DBLP TOC HTML
  └─ conf/{dblp_id}/{dblp_id}{year}.html
  └─ 전체 논문 제목 + DOI URL 추출

        ↓ KEYWORDS 필터링 (제목 기준)

Semantic Scholar API
  └─ 필터된 논문 제목으로 검색
  └─ abstract 첫 문장 추출

        ↓

data/cache.json  ←→  (증분 저장, 재실행 시 캐시 히트)

        ↓

README.md 자동 생성
  └─ 비율 요약표 (컨퍼런스 × 연도, 셀마다 섹션 링크)
  └─ 상세 논문 목록 (컨퍼런스별 → 연도별)
```

---

## 주의사항

- `data/cache.json`은 `.gitignore`에 포함되어 있습니다 (재생성 가능).
- `.env` 파일에는 API 키가 포함되므로 절대 커밋하지 마세요.
- DBLP에 proceedings가 아직 등록되지 않은 연도는 `0/0`으로 표시되며 README에서 `-`로 나타납니다.
- Semantic Scholar API 없이도 동작하지만 abstract 요약이 비어있을 수 있습니다.
