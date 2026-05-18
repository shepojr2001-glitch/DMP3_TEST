```mermaid
graph TD
    A([Main]) --> C[/user_input, hs_manager/]
    C -->|국내 분류사례 원문 검색| D[handle_domestic_case_lookup]
    C -->|해외 분류사례 원문 검색| E[handle_overseas_case_lookup]
    C -->|HS 해설서 원문 검색| F[get_hs_explanations]
    D -->|참고문서번호 직접 검색| G{ 패턴 매치 여부}
    D -->|키워드기반 문자열 검색| H[/user_input, top_k/]
    
    G --> |Y| GA[저장되어있는 JSON에서 CASE를 찾음]
    G --> |N| GB[사례를 찾을 수 없음 오류]

    H --> HA[JSON파일 내에서 CASE 별로 토큰 개수를 계산하여 상위 TOP_K개 만큼 추출] 
     
    E --> |참고문서번호검색| EA[JSON에서 케이스,국가를 찾음]
    E --> |HS코드검색| EB[해외HS코드분류데이터에서 해당 HS코드를 조회]
    E --> |키워드기반 문자열검색| EC[국내원문검색과 동일]

    F --> |knowledge/| FA[류,호,부 순서로 grouped_11_end.json 파일내 HS 코드에 대한 해설을취합]

