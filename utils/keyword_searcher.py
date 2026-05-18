"""
키워드 기반 HS 분류 사례 검색 엔진

이 모듈은 단순 키워드 매칭 방식으로 HS 분류 사례를 검색합니다.
- 참고문서번호 검색
- 키워드 토큰 기반 OR 검색
- HS 코드 직접 검색
"""

import re
from typing import List, Dict, Any


class KeywordCaseSearcher:
    """
    키워드 기반 HS 분류 사례 검색 클래스

    TF-IDF 검색과 달리 단순 문자열 매칭으로 사례를 검색합니다.
    참고문서번호 검색이나 특정 키워드 정확 매칭에 유용합니다.
    """

    def __init__(self, data_manager):
        """
        Args:
            data_manager: HSDataManager 인스턴스 (데이터 소스)
        """
        self.data_manager = data_manager

    def _tokenize_query(self, query: str) -> List[str]:
        """
        검색 쿼리를 토큰으로 분리하는 메서드 (검색 전용)

        Args:
            query: 검색 쿼리 문자열

        Returns:
            토큰 리스트 (최소 2글자 이상)
        """
        # 특수문자 제거 및 공백 기준 분리
        tokens = re.sub(r'[^\w\s]', ' ', query).split()
        # 길이 2 이상인 토큰만 반환 (중복 허용 - 빈도 계산에 사용)
        return [token.strip() for token in tokens if len(token.strip()) >= 2]

    def search_domestic_by_keyword(
        self,
        keyword: str,
        dict_word: list,
        top_k: int = 10,
        ignore_spaces: bool = False,
        min_tokens: int = 1
    ) -> List[Dict[str, Any]]:
        """
        고급 키워드 검색 (국내 분류사례)

        개선 사항:
        1. 토큰화: 검색어를 공백으로 분리
        2. OR 검색: 토큰 중 하나라도 매칭
        3. 띄어쓰기 무시: 공백 제거 후 검색 옵션
        4. 가중치 적용: 매칭된 토큰 개수로 순위 매김

        Args:
            keyword: 검색할 키워드 (여러 단어 가능)
            top_k: 반환할 최대 결과 개수
            ignore_spaces: True시 띄어쓰기 무시하고 검색
            min_tokens: 최소 매칭 토큰 수 (기본 1)

        Returns:
            검색 결과 리스트 (가중치 순으로 정렬)
        """
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'knowledge/HS위원회', 'knowledge/HS협의회'
        ]

        # 1. 토큰화
        
        #2026.05.13 추가 
        # 토큰화 된 단어와 용어사전에 있는단어를 매칭
        
        # tokens = self._tokenize_query(keyword)
        
        # if not tokens:
        #     return []
                
        
        # tokens_lower = [t.lower() for t in tokens]

        # # 결과를 점수와 함께 저장
        # scored_results = []
        
        """
        05.14
        토큰 검색 구조 수정
        1. 기존 단어를 토큰화
        2. 토큰도 없고 expansion_query도 없으면 그냥 리턴
        2-1. 질문이 "expansion_query" 이고 dict_word 만 있다면 -> expansion_query 사용        
        3. 토큰화된 단어와 용어사전 내 단어와 매칭
        4. 매칭된 단어가 있으면 vaild_token에 넣고 기존 토큰으로 대체(없으면 토큰 단어 그대로 사용)
        """
        tokens=[]
        tokens_lower=[]
        # 확장쿼리 값을 쓰지 않는다면
        if keyword != "expansion_query":
            tokens = self._tokenize_query(keyword)
            if dict_word:
                valid_token = [
                    token for token in tokens
                    if token in dict_word                
                ]
                if valid_token:
                   tokens = valid_token
            tokens_lower = [t.lower() for t in tokens]           
        # 확장 쿼리 값을 쓴다면
        elif keyword == "expansion_query" and dict_word:            
            tokens_lower = dict_word
            
        # 아니면            
        elif not tokens and not dict_word:
            return []
        print(keyword,tokens_lower)
        
        # tokens_lower = [t.lower() for t in tokens]
        
        # 결과를 점수와 함께 저장
        scored_results = []        
        """
        기존 -> '품목명 설명 분류근거'를 한 문장으로 엮어 토큰단어 존재여부 판단
        신규 -> ‘품목명’, ‘품목설명’, ‘분류근거’ 항목을 각각 분리하여 토큰 단어의 포함 여부를 판단하며,
                항목별로 서로 다른 가중치를 적용하여 점수를 산정
                 - 제품명에 토큰단어 있을 시 : 5점
                 - 품목설명에 토큰단어 있을 시 : 3점
                 - 분류근거에 토큰단어 있을 시 : 2점
        """
        # 결과를 점수와 함께 저장
        # scored_results = []        
        for source in domestic_sources:
            if source in self.data_manager.data:
                for item in self.data_manager.data[source]:
                    # 품목명, 설명, 분류근거에서 검색
                    searchable_texts = {
                        'product_name' : str(item.get('product_name', '')).lower(),
                        'description' : str(item.get('description', '')).lower(),
                        'decision_reason' : str(item.get('decision_reason', '')).lower()
                    }
                    matched_tokens = 0                    
                    for weight,searchable_text in searchable_texts.items():
                        weight_val = 0
                        
                        if weight == 'product_name': 
                            weight_val = 5                        
                        elif weight == 'description': 
                            weight_val = 3                        
                        elif weight == 'decision_reason': 
                            weight_val = 2                                                                        
                        
                        # 띄어쓰기 무시 옵션
                        if ignore_spaces :
                            searchable_text_no_space = searchable_text.replace(' ', '')                        
                        # 가중치 계산: 매칭된 토큰 개수 카운트                        
                        for token in tokens_lower:  
                            # if keyword == "expansion_query" : print(token)                                                  
                            if ignore_spaces:
                                token_no_space = token.replace(' ', '')
                                if token_no_space in searchable_text_no_space:
                                    matched_tokens += weight_val                                
                            else:                                
                                if token in searchable_text:
                                    matched_tokens += weight_val
                                
                    
                    
                    # OR 검색: 최소 토큰 수 이상 매칭되면 포함
                    if matched_tokens >= min_tokens:
                        scored_results.append((matched_tokens, item))
                                
        
        # 점수 기준 내림차순 정렬
        scored_results.sort(key=lambda x: x[0], reverse=True)
        print("scored_results")        
        # 상위 top_k개만 반환 (점수는 제외)
        # return [item for score, item in scored_results[:top_k]]
        return scored_results[:top_k]
    
    def find_domestic_case_by_id(self, ref_id: str) -> Dict[str, Any]:
        """
        참고문서번호로 국내 분류사례 검색

        Args:
            ref_id: 참고문서번호 (예: "품목분류2과-9433")

        Returns:
            해당 사례 딕셔너리 또는 None
        """
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'knowledge/HS위원회', 'knowledge/HS협의회'
        ]

        for source in domestic_sources:
            if source in self.data_manager.data:
                for item in self.data_manager.data[source]:
                    if item.get('reference_id') == ref_id:
                        return item
        return None

    def search_overseas_by_keyword(
        self,
        keyword: str,
        dict_word: list,
        top_k: int = 10,
        ignore_spaces: bool = False,
        min_tokens: int = 1
    ) -> List[Dict[str, Any]]:
        """
        고급 키워드 검색 (해외 분류사례)

        개선 사항:
        1. 토큰화: 검색어를 공백으로 분리
        2. OR 검색: 토큰 중 하나라도 매칭
        3. 띄어쓰기 무시: 공백 제거 후 검색 옵션
        4. 가중치 적용: 매칭된 토큰 개수로 순위 매김

        Args:
            keyword: 검색할 키워드 (여러 단어 가능)
            top_k: 반환할 최대 결과 개수
            ignore_spaces: True시 띄어쓰기 무시하고 검색
            min_tokens: 최소 매칭 토큰 수 (기본 1)

        Returns:
            검색 결과 리스트 (가중치 순으로 정렬)
        """
        '''기존 코드
        # 1. 토큰화
        tokens = self._tokenize_query(keyword)
        if not tokens:
            return []
        tokens_lower = [t.lower() for t in tokens]
        # 결과를 점수와 함께 저장
        scored_results = []
        '''
        tokens=[]
        tokens_lower=[]
        scored_results = []
        # 일반 키워드 검색(검색어 -> 토큰 저장)
        if keyword != "expansion_query":
            tokens = self._tokenize_query(keyword)
            if dict_word:
                valid_token = [
                    token for token in tokens
                    if token in dict_word                
                ]
                if valid_token:
                   tokens = valid_token
            tokens_lower = [t.lower() for t in tokens]
        # 확장 쿼리
        elif keyword == "expansion_query" and dict_word:            
            tokens_lower = [word.lower() for word in dict_word]
            
        elif not tokens and not dict_word:
            return []
        print(keyword,tokens_lower)
            
        for source in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if source in self.data_manager.data:
                for item in self.data_manager.data[source]:
                    """
                    기존 -> '품목명 설명 분류근거'를 한 문장으로 엮어 토큰단어 존재여부 판단
                    신규 -> ‘품목명’, ‘품목설명’, ‘분류근거’ 항목을 각각 분리하여 토큰 단어의 포함 여부를 판단하며,
                    항목별로 서로 다른 가중치를 적용하여 점수를 산정
                    - reply, keyword 에 토큰단어 있을 시 : 5점
                    - description 에 토큰단어 있을 시 : 1점
                    """
                    # keywords, 설명, reply에서 검색
                    searchable_texts = {
                        'reply' : str(item.get('reply', '')).lower(),
                        'description' : str(item.get('description', '')).lower(),
                        'keyword' : str(item.get('keyword', '')).lower()
                    }

                    # 가중치 계산
                    matched_tokens = 0
                    for weight,searchable_text in searchable_texts.items():
                        weight_val = 0
                        
                        if weight == 'reply': 
                            weight_val = 5                        
                        elif weight == 'description': 
                            weight_val = 1                        
                        elif weight == 'keyword': 
                            weight_val = 5  
                        # 띄어쓰기 무시 옵션
                        if ignore_spaces:
                            searchable_text_no_space = searchable_texts.replace(' ', '')
                        for token in tokens_lower:
                            if ignore_spaces:
                                token_no_space = token.replace(' ', '')
                                if token_no_space in searchable_text_no_space:
                                    matched_tokens += weight_val
                            else:
                                if token in searchable_text:
                                    matched_tokens += weight_val

                    # OR 검색: 최소 토큰 수 이상 매칭되면 포함
                    if matched_tokens >= min_tokens:
                        scored_results.append((matched_tokens, item))

        # 점수 기준 내림차순 정렬
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # 상위 top_k개만 반환 (점수는 제외)
        return scored_results[:top_k]

    def find_overseas_case_by_id(self, ref_id: str) -> Dict[str, Any]:
        """
        참고문서번호로 해외 분류사례 검색

        Args:
            ref_id: 참고문서번호 (예: "NY N338825")

        Returns:
            {'case': 사례 딕셔너리, 'country': 'US'/'EU'} 또는 None
        """
        for source in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if source in self.data_manager.data:
                for item in self.data_manager.data[source]:
                    if item.get('reference_id') == ref_id:
                        country = 'US' if 'us' in source else 'EU'
                        return {'case': item, 'country': country}
        return None

    def search_overseas_by_hs_code(self, hs_code: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        HS 코드로 해외 분류사례 검색

        Args:
            hs_code: HS 코드 (4-10자리, 예: "5515", "5515.12")
            top_k: 반환할 최대 결과 개수

        Returns:
            매칭되는 사례 리스트 (국가 정보 포함)
        """
        results = []

        for source in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if source in self.data_manager.data:
                country = 'US' if 'us' in source else 'EU'
                for item in self.data_manager.data[source]:
                    item_hs_code = item.get('hs_code', '')
                    # HS 코드 부분 매칭 (공백, 점, 하이픈 제거 후 비교)
                    if hs_code.replace('.', '').replace(' ', '') in item_hs_code.replace('.', '').replace(' ', '').replace('-', ''):
                        results.append({
                            'case': item,
                            'country': country
                        })
                        if len(results) >= top_k:
                            return results
        return results
