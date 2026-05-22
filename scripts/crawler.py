"""
DBLP TOC HTML로 컨퍼런스 논문 전체 목록을 수집하고,
키워드 필터링 후 Semantic Scholar로 abstract를 보완합니다.

DBLP TOC URL 패턴: https://dblp.uni-trier.de/db/conf/{dblp_id}/{toc_prefix}{year}.html
Semantic Scholar API: https://api.semanticscholar.org/api-docs/graph
"""

import requests
import time
import json
import re
import html
import warnings
from pathlib import Path
from config import CONFERENCES, KEYWORDS, START_YEAR, END_YEAR

CACHE_FILE = Path(__file__).parent.parent / 'data' / 'cache.json'
SS_BASE_URL = 'https://api.semanticscholar.org/graph/v1'
DBLP_BASE_URL = 'https://dblp.uni-trier.de/db/conf'

_RATE_WITH_KEY    = 10  # req/sec (Semantic Scholar - API 키 있을 때)
_RATE_WITHOUT_KEY =  1  # req/sec (Semantic Scholar - 공개)
_RATE_DBLP        =  1  # req/sec (DBLP HTML 요청 - 예의 바르게)

MAX_RETRIES  = 5
BASE_BACKOFF = 2.0

# ---------------------------------------------------------------------------
# SSL 설정 (기업 프록시 환경 대응)
# ---------------------------------------------------------------------------
_ssl_verify = True


def _get_ssl_verify() -> bool:
    return _ssl_verify


def _disable_ssl_verify():
    global _ssl_verify
    if _ssl_verify:
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        requests.packages.urllib3.disable_warnings()
        _ssl_verify = False
        print("[경고] SSL 인증서 검증을 비활성화했습니다 (프록시/자체서명 인증서 환경).")


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """요청 간격을 강제해 API 속도 제한을 준수합니다."""

    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second
        self._last = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last
        gap = self.min_interval - elapsed
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


# ---------------------------------------------------------------------------
# API 키 검증
# ---------------------------------------------------------------------------

def validate_api_key(api_key: str) -> bool:
    if not api_key or not api_key.strip():
        return False
    try:
        resp = _raw_get(
            f'{SS_BASE_URL}/paper/search',
            params={'query': 'recommendation', 'limit': 1, 'fields': 'title'},
            headers={'x-api-key': api_key},
        )
        if resp is None:
            return False
        if resp.status_code in (401, 403):
            print(f"[경고] API 키가 유효하지 않습니다 (HTTP {resp.status_code}). 공개 속도 제한으로 전환합니다.")
            return False
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[경고] API 키 검증 중 오류: {e}. 공개 속도 제한으로 전환합니다.")
        return False


def make_rate_limiter(api_key: str | None) -> tuple[RateLimiter, dict]:
    if api_key and validate_api_key(api_key):
        print(f"[OK] API 키 유효 -- {_RATE_WITH_KEY} req/sec 모드")
        return RateLimiter(_RATE_WITH_KEY), {'x-api-key': api_key}
    else:
        print(f"[INFO] 공개 모드 -- {_RATE_WITHOUT_KEY} req/sec 모드")
        return RateLimiter(_RATE_WITHOUT_KEY), {}


# ---------------------------------------------------------------------------
# HTTP 요청 (SSL 자동 재시도 포함)
# ---------------------------------------------------------------------------

def _raw_get(url: str, params: dict, headers: dict) -> requests.Response | None:
    try:
        return requests.get(url, params=params, headers=headers,
                            timeout=30, verify=_get_ssl_verify())
    except requests.exceptions.SSLError:
        _disable_ssl_verify()
        return requests.get(url, params=params, headers=headers,
                            timeout=30, verify=False)
    except requests.RequestException:
        raise


def _request_with_backoff(
    url: str,
    params: dict,
    headers: dict,
    rate_limiter: RateLimiter,
) -> dict | None:
    for attempt in range(MAX_RETRIES):
        rate_limiter.wait()
        try:
            resp = _raw_get(url, params=params, headers=headers)
            if resp is None:
                break

            if resp.status_code == 429:
                wait = BASE_BACKOFF ** (attempt + 1)
                print(f"    [429] 속도 초과 -- {wait:.0f}초 대기 후 재시도 ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            wait = BASE_BACKOFF ** (attempt + 1)
            print(f"    [오류] {e} -- {wait:.0f}초 후 재시도 ({attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)

    print(f"    [실패] {MAX_RETRIES}회 재시도 초과. 해당 요청 건너뜁니다.")
    return None


# ---------------------------------------------------------------------------
# DBLP TOC HTML 파싱
# ---------------------------------------------------------------------------

_DBLP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; RecSysResearch/1.0; mailto:researcher@gmail.com)',
    'Accept': 'text/html',
}


def _parse_dblp_html(html: str) -> list[dict]:
    """DBLP TOC HTML에서 inproceedings 논문 목록을 추출합니다."""
    entries = re.split(r'(?=<li class="entry inproceedings")', html)
    papers = []
    for entry in entries:
        if 'class="entry inproceedings"' not in entry:
            continue

        # 제목 추출
        title_m = re.search(r'<span class="title"[^>]*>(.*?)</span>', entry, re.DOTALL)
        if not title_m:
            continue
        title = html.unescape(re.sub(r'<[^>]+>', '', title_m.group(1)).strip().rstrip('.'))
        if not title:
            continue

        # URL: DOI 우선, 그다음 ee 링크
        doi_m = re.search(r'href="(https://doi\.org/[^"]+)"', entry)
        if doi_m:
            url = doi_m.group(1)
        else:
            ee_m = re.search(r'class="ee"[^>]*>.*?href="([^"]+)"', entry, re.DOTALL)
            url = ee_m.group(1) if ee_m else None

        papers.append({'title': title, 'url': url})

    return papers


def fetch_dblp_papers(dblp_id: str, toc_prefix: str, year: int, dblp_rl: RateLimiter) -> list[dict]:
    """
    DBLP TOC HTML에서 특정 컨퍼런스·연도의 전체 논문을 수집합니다.
    URL: https://dblp.uni-trier.de/db/conf/{dblp_id}/{toc_prefix}{year}.html
    """
    url = f'{DBLP_BASE_URL}/{dblp_id}/{toc_prefix}{year}.html'

    for attempt in range(MAX_RETRIES):
        dblp_rl.wait()
        try:
            resp = _raw_get(url, params={}, headers=_DBLP_HEADERS)
            if resp is None:
                break

            if resp.status_code == 404:
                print(f"    [DBLP] {dblp_id}/{toc_prefix}{year}.html -- 404 (해당 연도 proceedings 없음)")
                return []

            if resp.status_code == 429:
                wait = BASE_BACKOFF ** (attempt + 1)
                print(f"    [429] DBLP 속도 초과 -- {wait:.0f}초 대기")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                wait = BASE_BACKOFF ** (attempt + 1)
                print(f"    [오류] DBLP HTTP {resp.status_code} -- {wait:.0f}초 후 재시도")
                time.sleep(wait)
                continue

            papers = _parse_dblp_html(resp.text)
            return papers

        except requests.RequestException as e:
            wait = BASE_BACKOFF ** (attempt + 1)
            print(f"    [오류] DBLP 요청 실패: {e} -- {wait:.0f}초 후 재시도")
            time.sleep(wait)

    print(f"    [실패] DBLP {dblp_id} {year} 수집 실패")
    return []


# ---------------------------------------------------------------------------
# Semantic Scholar: abstract 보완
# ---------------------------------------------------------------------------

def fetch_ss_abstract(title: str, rate_limiter: RateLimiter, headers: dict) -> str | None:
    """제목으로 Semantic Scholar를 검색해 abstract 첫 문장을 반환합니다."""
    data = _request_with_backoff(
        url=f'{SS_BASE_URL}/paper/search',
        params={'query': title, 'fields': 'title,abstract', 'limit': 1},
        headers=headers,
        rate_limiter=rate_limiter,
    )
    if not data:
        return None
    papers = data.get('data', [])
    if not papers:
        return None
    return get_first_sentence(papers[0].get('abstract'))


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def is_recommendation_paper(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in KEYWORDS)


def get_first_sentence(abstract: str | None) -> str | None:
    if not abstract:
        return None
    sentences = re.split(r'(?<=[.!?])\s+', abstract.strip())
    first = sentences[0].strip() if sentences else ''
    return (first[:247] + '...') if len(first) > 250 else first or None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 수집 로직
# ---------------------------------------------------------------------------

def crawl_conference_year(
    conf_key: str,
    year: int,
    cache: dict,
    rate_limiter: RateLimiter,
    headers: dict,
    dblp_rl: RateLimiter,
    force: bool = False,
) -> dict:
    """
    DBLP TOC HTML로 컨퍼런스·연도 전체 논문을 수집하고,
    키워드 필터 후 Semantic Scholar로 abstract를 보완합니다.
    """
    cache_key = f"{conf_key}_{year}"

    if cache_key in cache and not force:
        d = cache[cache_key]
        print(f"  [캐시] {conf_key} {year}: {d['rec_count']}/{d['total_papers']} ({d['ratio']:.1%})")
        return cache[cache_key]

    conf_cfg = CONFERENCES[conf_key]
    dblp_id = conf_cfg['dblp_id']
    toc_prefix = conf_cfg.get('dblp_toc_prefix', dblp_id)

    print(f"  [수집] {conf_key} {year} -- DBLP TOC 수집 중...")
    papers_list = fetch_dblp_papers(dblp_id, toc_prefix, year, dblp_rl)
    total_count = len(papers_list)
    print(f"  [DBLP] {conf_key} {year}: 총 {total_count}편 수집")

    rec_papers = []
    for paper in papers_list:
        title = paper['title']
        if not is_recommendation_paper(title):
            continue

        url = paper.get('url')
        print(f"    [SS] abstract 조회: {title[:55]}...")
        summary = fetch_ss_abstract(title, rate_limiter, headers)

        rec_papers.append({
            'title': title,
            'url': url,
            'summary': summary,
            'year': year,
        })

    rec_papers.sort(key=lambda x: x['title'].lower())

    result = {
        'conference': conf_key,
        'year': year,
        'total_papers': total_count,
        'rec_papers': rec_papers,
        'rec_count': len(rec_papers),
        'ratio': len(rec_papers) / total_count if total_count else 0.0,
    }

    cache[cache_key] = result
    print(f"  [완료] {conf_key} {year}: {result['rec_count']}/{result['total_papers']} ({result['ratio']:.1%})")
    return result


def crawl_all(
    api_key: str | None = None,
    force_refresh: set[tuple] | None = None,
) -> dict:
    rate_limiter, headers = make_rate_limiter(api_key)
    dblp_rl = RateLimiter(_RATE_DBLP)
    force_set = force_refresh or set()
    cache = load_cache()

    for conf_key in CONFERENCES:
        for year in range(START_YEAR, END_YEAR + 1):
            crawl_conference_year(
                conf_key, year, cache,
                rate_limiter=rate_limiter,
                headers=headers,
                dblp_rl=dblp_rl,
                force=(conf_key, year) in force_set,
            )
            save_cache(cache)

    return cache
