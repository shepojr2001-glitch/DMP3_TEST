import json
import re
import os
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv
from .text_utils import clean_text, general_explanation
from .api_retry import retry_on_api_error

# API client는 main.py에서 파라미터로 전달받음

def lookup_hscode(hs_code, json_file):
    """HS 코드에 대한 해설 정보를 조회하는 함수"""
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 각 설명 유형별 초기값 설정
        part_explanation = {"text": "해당 부에 대한 설명을 찾을 수 없습니다."}
        chapter_explanation = {"text": "해당 류에 대한 설명을 찾을 수 없습니다."}
        sub_explanation = {"text": "해당 호에 대한 설명을 찾을 수 없습니다."}

        # 1) 류(類) key: "제00류"
        chapter_key = f"제{int(hs_code[:2])}류"
        chapter_explanation = next((g for g in data if g.get('header2') == chapter_key), chapter_explanation)

        # 2) 호 key: "00.00" (4자리까지만 사용)
        hs_4digit = hs_code[:4]  # 4자리까지만 추출
        sub_key = f"{hs_4digit[:2]}.{hs_4digit[2:]}"
        sub_explanation = next((g for g in data if g.get('header2') == sub_key), sub_explanation)

        # 3) 부(部) key: "제00부"
        part_key = chapter_explanation.get('header1') if chapter_explanation else None
        part_explanation = next((g for g in data if (g.get('header1') == part_key)&(re.sub(r'제\s*(\d+)\s*부', r'제\1부', g.get('header1')) == part_key)), None)

        return part_explanation, chapter_explanation, sub_explanation

    except Exception as e:
        print(f"HS 코드 조회 오류: {e}")
        return ({"text": "오류가 발생했습니다."}, {"text": "오류가 발생했습니다."}, {"text": "오류가 발생했습니다."})

def get_hs_explanations(hs_codes):
    """여러 HS 코드에 대한 해설을 취합하는 함수 (마크다운 형식)"""
    all_explanations = ""
    for hs_code in hs_codes:
        explanation, type_explanation, number_explanation = lookup_hscode(hs_code, 'knowledge/grouped_11_end.json')

        if explanation and type_explanation and number_explanation:
            all_explanations += f"\n\n# HS 코드 {hs_code} 해설\n\n"
            all_explanations += f"## 📋 해설서 통칙\n\n"

            # 통칙 내용을 리스트 형태로 정리
            if general_explanation:
                for i, rule in enumerate(general_explanation[:5], 1):  # 처음 5개만 표시
                    all_explanations += f"### 통칙 {i}\n{rule}\n\n"

            all_explanations += f"## 📂 부(部) 해설\n\n{explanation['text']}\n\n"
            all_explanations += f"## 📚 류(類) 해설\n\n{type_explanation['text']}\n\n"
            all_explanations += f"## 📝 호(號) 해설\n\n{number_explanation['text']}\n\n"
            all_explanations += "---\n"  # 구분선 추가

    return all_explanations

def get_tariff_info_for_codes(hs_codes):
    """HS코드들에 대한 품목분류표 정보 수집"""
    tariff_info = {}

    try:
        with open('knowledge/hstable.json', 'r', encoding='utf-8') as f:
            tariff_data = json.load(f)

        for code in hs_codes:
            # 4자리 HS코드로 매칭 (예: 3923 또는 39.23)
            code_4digit = code[:4] if len(code) >= 4 else code
            code_with_dot = f"{code_4digit[:2]}.{code_4digit[2:]}"

            for item in tariff_data:
                item_code = item.get('품목번호', '')
                if item_code.startswith(code_4digit) or item_code.startswith(code_with_dot):
                    tariff_info[code] = {
                        'korean_name': item.get('한글품명', ''),
                        'english_name': item.get('영문품명', ''),
                        'full_code': item_code
                    }
                    break
    except Exception as e:
        print(f"Tariff table loading error: {e}")

    return tariff_info

def get_manual_info_for_codes(hs_codes, logger, client):
    """HS코드들에 대한 해설서 정보 수집 및 요약"""
    manual_info = {}

    for code in hs_codes:
        try:
            # lookup_hscode 함수 재사용
            part_exp, chapter_exp, sub_exp = lookup_hscode(code, 'knowledge/grouped_11_end.json')

            # 해설서 내용 조합
            full_content = ""
            if part_exp and part_exp.get('text'):
                full_content += f"부 해설: {part_exp['text']}\n\n"
            if chapter_exp and chapter_exp.get('text'):
                full_content += f"류 해설: {chapter_exp['text']}\n\n"
            if sub_exp and sub_exp.get('text'):
                full_content += f"호 해설: {sub_exp['text']}\n\n"

            # 1000자 초과 시 요약
            if len(full_content) > 1000:
                logger.log_actual("AI", f"Summarizing manual content for HS{code}...")
                summary_prompt = f"""다음 HS 해설서 내용을 1000자 이내로 핵심 내용만 요약해주세요:

HS코드: {code}
해설서 내용:
{full_content}

요약 시 포함할 내용:
- 주요 품목 범위
- 포함/제외 품목
- 분류 기준
- 핵심 특징

간결하고 정확하게 요약해주세요."""

                try:
                    # 재시도 로직 적용
                    @retry_on_api_error(max_retries=3, initial_delay=0.5)
                    def _summary_api_call():
                        return client.models.generate_content(
                            model="gemini-2.5-flash-lite",
                            contents=summary_prompt
                        )

                    summary_response = _summary_api_call()
                    manual_info[code] = {
                        'content': clean_text(summary_response.text),
                        'summary_used': True
                    }
                    logger.log_actual("SUCCESS", f"HS{code} manual summarized", f"{len(manual_info[code]['content'])} chars")

                except APIError as e:
                    # API 에러 (재시도 후에도 실패)
                    logger.log_actual("ERROR", f"HS{code} API error after retries (code: {e.code})", e.message)
                    manual_info[code] = {
                        'content': full_content[:1000] + "...",
                        'summary_used': False
                    }

                except Exception as e:
                    logger.log_actual("ERROR", f"HS{code} summary failed: {str(e)}")
                    manual_info[code] = {
                        'content': full_content[:1000] + "...",
                        'summary_used': False
                    }
            else:
                manual_info[code] = {
                    'content': full_content,
                    'summary_used': False
                }

        except Exception as e:
            logger.log_actual("ERROR", f"HS{code} manual loading failed: {str(e)}")
            manual_info[code] = {
                'content': "해설서 정보를 찾을 수 없습니다.",
                'summary_used': False
            }

    return manual_info

def prepare_general_rules():
    """HS 분류 통칙 준비"""
    try:
        with open('knowledge/통칙_grouped.json', 'r', encoding='utf-8') as f:
            rules_data = json.load(f)

        rules_text = "HS 분류 통칙:\n\n"
        for i, rule in enumerate(rules_data[:6], 1):  # 통칙 1~6
            rules_text += f"통칙 {i}: {rule.get('text', '')}\n\n"

        return rules_text
    except Exception as e:
        return "통칙 정보를 로드할 수 없습니다."

def analyze_user_provided_codes(user_input, hs_codes, tariff_info, manual_info, general_rules, context, client):
    """사용자 제시 HS코드들에 대한 최종 AI 분석"""

    # HS 해설서 분석 전용 맞춤형 프롬프트
    manual_analysis_context = """당신은 HS 해설서 및 품목분류표 전문 분석가입니다.

역할과 목표:
- 사용자가 제시한 여러 HS코드 중 가장 적합한 코드 선택
- 품목분류표 품명과 HS 해설서 내용을 기반으로 한 체계적 비교
- HS 통칙을 적용한 논리적 분류 근거 제시

분석 방법:
- **각 코드별 개별 분석**: 품목분류표 품명과 해설서 내용 검토
- **비교 분석**: 사용자 물품과의 적합성 비교
- **통칙 적용**: 해당되는 HS 통칙과 적용 근거
- **최종 추천**: 가장 적합한 HS코드와 명확한 선택 이유

주의사항:
- 답변 시 반드시 출처를 명시하세요 (예: "제39류 HS 해설서에 따르면...", "호 3923 해설서에 따르면...", "통칙 제3호에 따르면...")
- 사용자가 자료에 없는 내용을 물어볼 경우, 반드시 "해당 정보는 HS 해설서에 없습니다" 또는 "확인된 자료가 없습니다"라고 답변하세요

답변 구성요소:
1. **최적 HS코드 추천**: 가장 적합한 코드와 선택 이유
2. **각 코드별 분석**: 개별 평가 및 적합성 판단
3. **통칙 적용**: 관련 통칙과 적용 근거
4. **최종 결론**: 추천 코드와 주의사항

사용자가 제시한 HS코드를 중심으로 정확하고 전문적인 비교 분석을 제공해주세요."""

    # 분석용 프롬프트 구성
    analysis_prompt = f"""{manual_analysis_context}

{general_rules}

사용자가 제시한 HS코드별 상세 정보:

"""

    for code in hs_codes:
        analysis_prompt += f"""
=== HS코드 {code} ===
품목분류표 정보:
- 국문품명: {tariff_info.get(code, {}).get('korean_name', 'N/A')}
- 영문품명: {tariff_info.get(code, {}).get('english_name', 'N/A')}

해설서 정보:
{manual_info.get(code, {}).get('content', 'N/A')}

"""

    analysis_prompt += f"""
사용자 질문: {user_input}

위의 HS 분류 통칙과 각 HS코드별 상세 정보를 바탕으로 다음을 포함하여 답변해주세요:

1. **최적 HS코드 추천**
   - 사용자가 제시한 HS코드 중 가장 적합한 코드 선택
   - 선택 이유와 근거 제시

2. **각 코드별 분석**
   - 각 HS코드가 사용자 물품에 적합한지 평가
   - 품목분류표 품명과 해설서 내용 기반 분석

3. **통칙 적용**
   - 해당되는 HS 통칙과 적용 근거
   - 분류 시 고려사항

4. **최종 결론**
   - 추천 HS코드와 분류 근거
   - 주의사항 및 추가 고려사항

전문적이면서도 이해하기 쉽게 답변해주세요."""

    # Gemini AI 분석 수행
    try:
        # 재시도 로직 적용
        @retry_on_api_error(max_retries=3, initial_delay=0.5)
        def _analysis_api_call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=analysis_prompt
            )

        response = _analysis_api_call()
        return clean_text(response.text)

    except APIError as e:
        # API 에러 (재시도 후에도 실패)
        error_msg = f"Gemini API 오류가 발생했습니다 (3회 재시도 후 실패).\n\n**오류 코드**: {e.code}\n**오류 메시지**: {e.message}\n\n"

        if e.code == 503:
            error_msg += "**해결 방법**: API 서버가 지속적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요."
        elif e.code == 429:
            error_msg += "**해결 방법**: API 사용량 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
        elif e.code == 404:
            error_msg += "**해결 방법**: 요청한 모델을 찾을 수 없습니다."
        elif e.code == 400:
            error_msg += "**해결 방법**: 잘못된 요청입니다. 입력 내용을 확인해주세요."
        else:
            error_msg += "**해결 방법**: 문제가 지속되면 관리자에게 문의해주세요."

        return error_msg

    except Exception as e:
        return f"처리 중 예상치 못한 오류가 발생했습니다.\n\n**오류 내용**: {str(e)}\n\n관리자에게 문의해주세요."