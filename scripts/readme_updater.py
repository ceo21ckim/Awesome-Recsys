"""
cache.json을 읽어 README.md(요약표+링크)와 papers/{conf}.md(상세 논문 목록)를 생성합니다.
"""

import json
from pathlib import Path
from datetime import datetime
from config import CONFERENCES, START_YEAR, KEYWORDS, TOPIC_LABEL

CACHE_FILE  = Path(__file__).parent.parent / 'data' / 'cache.json'
README_FILE = Path(__file__).parent.parent / 'README.md'
PAPERS_DIR  = Path(__file__).parent.parent / 'papers'

SECTION_START = '<!-- TREND_ANALYSIS_START -->'
SECTION_END   = '<!-- TREND_ANALYSIS_END -->'

CONF_DESC = {
    'RecSys':  'ACM Conference on Recommender Systems',
    'SIGIR':   'ACM SIGIR — Information Retrieval',
    'KDD':     'ACM SIGKDD — Knowledge Discovery & Data Mining',
    'WSDM':    'ACM WSDM — Web Search and Data Mining',
    'CIKM':    'ACM CIKM — Information and Knowledge Management',
    'WWW':     'The Web Conference (WWW)',
    'IJCAI':   'IJCAI — Artificial Intelligence',
    'ICML':    'ICML — Machine Learning',
    'AAAI':    'AAAI — Artificial Intelligence',
    'NeurIPS': 'NeurIPS — Neural Information Processing Systems',
}


def load_cache():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"캐시 파일이 없습니다. 먼저 main.py를 실행하세요: {CACHE_FILE}")
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_available_years(cache):
    years = set()
    for key in cache:
        parts = key.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            years.add(int(parts[1]))
    return sorted(years)


def anchor(text):
    """GitHub Markdown 앵커 슬러그 생성."""
    return text.lower().replace(' ', '-').replace('/', '')


# ---------------------------------------------------------------------------
# README 생성
# ---------------------------------------------------------------------------

def make_toc(all_years, cache):
    lines = ['## 목차\n']
    for conf_key in CONFERENCES:
        desc = CONF_DESC.get(conf_key, '')
        year_links = []
        for year in sorted(all_years, reverse=True):
            data = cache.get(f"{conf_key}_{year}")
            if data and data['total_papers'] > 0:
                anch = anchor(f"{conf_key} {year}")
                year_links.append(f"[{year}](papers/{conf_key}.md#{anch})")
        year_str = ' · '.join(year_links) if year_links else '*데이터 없음*'
        lines.append(f"- **[{conf_key}](papers/{conf_key}.md)** — {desc}")
        lines.append(f"  {year_str}")
    return '\n'.join(lines)


def make_ratio_table(cache, all_years):
    header  = '| 컨퍼런스 | ' + ' | '.join(str(y) for y in all_years) + ' |'
    divider = '|:---:|' + '|'.join([':---:'] * len(all_years)) + '|'
    rows = [header, divider]

    for conf_key in CONFERENCES:
        cells = []
        for year in all_years:
            data = cache.get(f"{conf_key}_{year}")
            if data and data['total_papers'] > 0:
                rec   = data['rec_count']
                total = data['total_papers']
                pct   = data['ratio'] * 100
                label = f"{rec}/{total} ({pct:.0f}%)"
                anch  = anchor(f"{conf_key} {year}")
                cells.append(f"[{label}](papers/{conf_key}.md#{anch})")
            else:
                cells.append('-')
        rows.append(f"| **[{conf_key}](papers/{conf_key}.md)** | " + ' | '.join(cells) + ' |')

    return '\n'.join(rows)


def generate_section(cache):
    all_years = get_available_years(cache)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    total_rec = sum(v.get('rec_count', 0) for v in cache.values())
    total_all = sum(v.get('total_papers', 0) for v in cache.values())
    pct_all   = total_rec / total_all * 100 if total_all else 0

    kw_display = ' · '.join(f'`{kw.capitalize()}`' for kw in KEYWORDS)

    toc         = make_toc(all_years, cache)
    ratio_table = make_ratio_table(cache, all_years)

    return f"""{SECTION_START}
# {TOPIC_LABEL} Papers at Top CS Conferences

> 자동 생성 | 마지막 업데이트: **{now}**
> 데이터 출처: [DBLP](https://dblp.org/) (논문 목록) + [Semantic Scholar](https://api.semanticscholar.org/) (요약)
>
> 분석 범위: {START_YEAR}년 이후 | 키워드: {kw_display} (제목 기준)
>
> 총 수집: **{total_all:,}편** 중 {TOPIC_LABEL} 관련 **{total_rec:,}편** ({pct_all:.1f}%)

---

{toc}

---

## 비율 요약표

{ratio_table}

> `-` : 해당 연도 proceedings가 아직 DBLP에 등록되지 않음
> 각 셀을 클릭하면 해당 컨퍼런스·연도의 상세 논문 목록으로 이동합니다.

{SECTION_END}"""


def update_readme(section_content):
    if README_FILE.exists():
        original = README_FILE.read_text(encoding='utf-8')
        if SECTION_START in original and SECTION_END in original:
            print("[INFO] 기존 트렌드 섹션을 교체합니다.")
        else:
            print("[INFO] README.md를 새로 작성합니다.")
    else:
        print("[INFO] README.md를 생성합니다.")

    README_FILE.write_text(section_content, encoding='utf-8')
    print("[완료] README.md 업데이트 완료!")


# ---------------------------------------------------------------------------
# papers/{conf}.md 생성
# ---------------------------------------------------------------------------

def make_conference_file(cache, conf_key, all_years, kw_inline):
    desc = CONF_DESC.get(conf_key, '')
    lines = [
        f"# {TOPIC_LABEL} Papers — {conf_key}",
        "",
        f"> {desc}",
        f"> [← README로 돌아가기](../README.md)",
        "",
        "---",
        "",
    ]

    for year in sorted(all_years, reverse=True):
        data = cache.get(f"{conf_key}_{year}")
        if not data or data['total_papers'] == 0:
            continue

        if not data.get('rec_papers'):
            lines += [
                f"### {conf_key} {year}",
                f"> 전체 {data['total_papers']}편 중 키워드 관련 논문 없음",
                "",
                "---",
                "",
            ]
            continue

        total     = data['total_papers']
        rec_count = data['rec_count']
        pct       = data['ratio'] * 100

        lines += [
            f"### {conf_key} {year}",
            f"> 전체 {total}편 중 **{rec_count}편** ({pct:.1f}%) — 키워드: {kw_inline}",
            "",
        ]
        for p in data['rec_papers']:
            title   = p['title']
            url     = p.get('url')
            summary = p.get('summary', '')
            line = f"- [{title}]({url})" if url else f"- {title}"
            if summary:
                line += f"\n  > {summary}"
            lines.append(line)
        lines += ["", "---", ""]

    return '\n'.join(lines)


def generate_conference_files(cache, all_years, kw_inline):
    PAPERS_DIR.mkdir(exist_ok=True)
    for conf_key in CONFERENCES:
        content = make_conference_file(cache, conf_key, all_years, kw_inline)
        path = PAPERS_DIR / f"{conf_key}.md"
        path.write_text(content, encoding='utf-8')
        print(f"[완료] papers/{conf_key}.md 생성")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def run():
    cache = load_cache()
    all_years = get_available_years(cache)
    kw_inline = ' / '.join(f'`{kw}`' for kw in KEYWORDS)

    generate_conference_files(cache, all_years, kw_inline)
    section = generate_section(cache)
    update_readme(section)


if __name__ == '__main__':
    run()
