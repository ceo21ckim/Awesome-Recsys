from datetime import datetime

START_YEAR = 2022
END_YEAR = datetime.now().year

# dblp_id: DBLP stream 경로 (conf/<id>/)
# ss_venue: Semantic Scholar abstract 보완 검색용 venue 키워드
CONFERENCES = {
    'RecSys': {
        'display_name': 'RecSys',
        'dblp_id': 'recsys',
        'full_name': 'ACM Conference on Recommender Systems',
    },
    'SIGIR': {
        'display_name': 'SIGIR',
        'dblp_id': 'sigir',
        'full_name': 'ACM SIGIR Conference on Research and Development in Information Retrieval',
    },
    'KDD': {
        'display_name': 'KDD',
        'dblp_id': 'kdd',
        'full_name': 'ACM SIGKDD Conference on Knowledge Discovery and Data Mining',
    },
    'WSDM': {
        'display_name': 'WSDM',
        'dblp_id': 'wsdm',
        'full_name': 'ACM International Conference on Web Search and Data Mining',
    },
    'CIKM': {
        'display_name': 'CIKM',
        'dblp_id': 'cikm',
        'full_name': 'ACM International Conference on Information and Knowledge Management',
    },
    'WWW': {
        'display_name': 'WWW',
        'dblp_id': 'www',
        'full_name': 'The Web Conference (formerly WWW)',
    },
    'IJCAI': {
        'display_name': 'IJCAI',
        'dblp_id': 'ijcai',
        'full_name': 'International Joint Conference on Artificial Intelligence',
    },
    'ICML': {
        'display_name': 'ICML',
        'dblp_id': 'icml',
        'full_name': 'International Conference on Machine Learning',
    },
    'AAAI': {
        'display_name': 'AAAI',
        'dblp_id': 'aaai',
        'full_name': 'AAAI Conference on Artificial Intelligence',
    },
    'NeurIPS': {
        'display_name': 'NeurIPS',
        'dblp_id': 'nips',
        'dblp_toc_prefix': 'neurips',  # URL 파일명이 dblp_id와 다름
        'full_name': 'Conference on Neural Information Processing Systems',
    },
}

# -----------------------------------------------------------------------
# 검색 설정 — 이 두 값을 바꾸면 다른 토픽으로 전환됩니다
# -----------------------------------------------------------------------

# 논문 제목에서 찾을 키워드 (소문자, 부분 일치)
# 예) LLM 추적: ['llm', 'large language model', 'foundation model']
# 예) GNN 추적: ['graph neural network', 'graph convolutional']
KEYWORDS = ['recommendation', 'recommender', 'recommend']

# README 제목·요약에 사용할 짧은 토픽 레이블
# 예) 'LLM', 'Graph Neural Network', 'Multimodal'
TOPIC_LABEL = 'Recommendation'
