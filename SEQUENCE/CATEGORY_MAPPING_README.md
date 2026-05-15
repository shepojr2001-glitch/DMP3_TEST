graph TD
    A([Main]) --> B[/user_input, hs_manager/]
    B --> C{CATEGORY_MAPPING}
    C -->|웹검색| D[handle_web_search]
    C -->|국내사례기반 HS추천| E[handle_hs_classification_cases]
    C -->|해외사례기반 HS추천| F[handle_overseas_hs]
    C -->|국내 분류사례 원문 검색| G[handle_domestic_case_lookup]
    C -->|해외 분류사례 원문 검색| H[handle_overseas_case_lookup]
    C -->|HS 해설서 분석| I[handle_hs_manual_with_user_codes]
    C -->|HS 해설서 원문 검색| J[handle_overseas_case_lookup]

    
    