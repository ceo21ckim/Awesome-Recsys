"""
사용법:
  python main.py                          # 캐시 없는 데이터만 수집 + README 업데이트
  python main.py --refresh SIGIR 2024    # 특정 컨퍼런스/연도 강제 재수집
  python main.py --readme-only           # 수집 없이 README만 업데이트
  python main.py --conf RecSys SIGIR     # 특정 컨퍼런스만 수집
  python main.py --year 2024 2025        # 특정 연도만 수집
"""

import argparse
import os
import sys
from pathlib import Path

# 파일/파이프로 실행할 때도 로그가 실시간으로 보이도록
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드 (없으면 무시)
load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent))

from crawler import crawl_conference_year, load_cache, save_cache, make_rate_limiter, RateLimiter, _RATE_DBLP
from readme_updater import run as update_readme
from config import CONFERENCES, START_YEAR, END_YEAR


def parse_args():
    parser = argparse.ArgumentParser(description='RecSys Trend Analyzer')
    parser.add_argument(
        '--refresh', nargs=2, metavar=('CONF', 'YEAR'),
        action='append', default=[],
        help='특정 컨퍼런스/연도를 강제 재수집. 예: --refresh SIGIR 2024'
    )
    parser.add_argument(
        '--readme-only', action='store_true',
        help='데이터 수집 없이 README만 업데이트'
    )
    parser.add_argument(
        '--conf', nargs='+', metavar='CONF',
        help='특정 컨퍼런스만 수집. 예: --conf RecSys SIGIR KDD'
    )
    parser.add_argument(
        '--year', nargs='+', type=int, metavar='YEAR',
        help='특정 연도만 수집. 예: --year 2024 2025'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')

    if args.readme_only:
        print("[INFO] README만 업데이트합니다...")
        update_readme()
        return

    if not api_key:
        print("[INFO] .env에 SEMANTIC_SCHOLAR_API_KEY가 없습니다.")
        print("       무료 API 키: https://www.semanticscholar.org/product/api")
        print("       키 없이도 동작하지만 속도가 10배 느립니다.\n")

    # API 키 검증 + 속도 제한 결정 (1회)
    rate_limiter, headers = make_rate_limiter(api_key)
    dblp_rl = RateLimiter(_RATE_DBLP)

    # 강제 재수집 목록 파싱
    force_refresh = set()
    for conf, year_str in args.refresh:
        if conf not in CONFERENCES:
            print(f"[오류] 알 수 없는 컨퍼런스: {conf}. 선택 가능: {list(CONFERENCES.keys())}")
            sys.exit(1)
        force_refresh.add((conf, int(year_str)))

    # 수집 대상 결정
    target_confs = args.conf if args.conf else list(CONFERENCES.keys())
    target_years = args.year if args.year else list(range(START_YEAR, END_YEAR + 1))

    for conf in target_confs:
        if conf not in CONFERENCES:
            print(f"[오류] 알 수 없는 컨퍼런스: {conf}")
            sys.exit(1)

    print(f"\n[수집 대상] {target_confs}")
    print(f"[대상 연도] {target_years}")
    if force_refresh:
        print(f"[강제 재수집] {sorted(force_refresh)}")
    print()

    # 수집 실행
    cache = load_cache()
    total = len(target_confs) * len(target_years)
    done = 0

    for conf_key in target_confs:
        for year in target_years:
            done += 1
            print(f"[{done}/{total}] {conf_key} {year}")
            crawl_conference_year(
                conf_key, year, cache,
                rate_limiter=rate_limiter,
                headers=headers,
                dblp_rl=dblp_rl,
                force=(conf_key, year) in force_refresh,
            )
            save_cache(cache)

    print("\n[INFO] README.md 업데이트 중...")
    update_readme()
    print("\n[완료]")


if __name__ == '__main__':
    main()
