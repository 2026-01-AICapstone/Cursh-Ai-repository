import json
import re
import pandas as pd
from deep_translator import GoogleTranslator


# --- 깨진 JSONL 줄을 자동으로 복구하는 함수 ---
def fix_broken_line(line):
    fixed_line = line

    # 1. "risk_area" 키가 중복으로 꼬인 경우 수정
    # (예: "risk_area": "Suicide "risk_area": "Suicide & Self-Harm" Self-Harm")
    # -> "risk_area": "Suicide & Self-Harm" 으로 복구
    fixed_line = re.sub(
        r'"risk_area":\s*"[^"]*"\s*risk_area":\s*"([^"]*)"\s*[^"]*"',
        r'"risk_area": "\1"',
        fixed_line
    )

    try:
        json.loads(fixed_line)
        return fixed_line
    except json.JSONDecodeError:
        pass

    # 2. "question" 필드 내용 안에 이스케이프 되지 않은 쌍따옴표(")가 있는 경우 수정
    # (문장 안의 "를 \" 로 변환하여 JSON 문법 오류 해결)
    match = re.search(r'("question":\s*")(.*?)("\s*,\s*"(original_question|source|risk_area)")', fixed_line, re.DOTALL)
    if match:
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)

        # 내부 따옴표 이스케이프 처리
        content = content.replace('\\"', '@@ESCAPED@@')
        content = content.replace('"', '\\"')
        content = content.replace('@@ESCAPED@@', '\\"')

        fixed_line = fixed_line[:match.start()] + prefix + content + suffix + fixed_line[match.end():]

    try:
        json.loads(fixed_line)
        return fixed_line
    except json.JSONDecodeError:
        return None  # 복구 실패 시 None 반환


# --- 메인 실행 함수 ---
def process_jsonl_to_excel(input_filename, fixed_jsonl_filename, output_excel_filename):
    extracted_data = []
    fixed_lines_count = 0
    failed_lines_count = 0

    print(f"📥 JSONL 파일('{input_filename}')을 읽고 복구하는 중...")

    # 1. 파일 읽기 및 자동 복구 후 새로운 JSONL 파일로 저장
    with open(input_filename, 'r', encoding='utf-8') as f_in, \
            open(fixed_jsonl_filename, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # 정상적인 JSON인지 테스트
                item = json.loads(line)
                f_out.write(line + '\n')  # 정상 데이터는 그대로 새 파일에 저장
            except json.JSONDecodeError:
                # 깨진 데이터라면 복구 시도
                repaired_line = fix_broken_line(line)
                if repaired_line:
                    item = json.loads(repaired_line)
                    f_out.write(repaired_line + '\n')  # 복구된 데이터를 새 파일에 저장
                    fixed_lines_count += 1
                    print(f"   [복구 완료] {line_num}번째 줄의 깨진 데이터를 수정했습니다.")
                else:
                    failed_lines_count += 1
                    print(f"   [복구 실패] {line_num}번째 줄은 복구할 수 없을 정도로 손상되었습니다.")
                    continue

            # 필요한 필드만 추출
            extracted_data.append({
                'risk_area': item.get('risk_area', 'Unknown'),
                'question': item.get('question', '')
            })

    print(f"\n✅ 데이터 추출 완료! (총 {len(extracted_data)}개 추출, 복구 성공: {fixed_lines_count}개, 복구 실패: {failed_lines_count}개)")
    print(f"💾 복구된 깔끔한 데이터가 '{fixed_jsonl_filename}' 파일로 저장되었습니다.\n")

    if not extracted_data:
        print("추출할 데이터가 없습니다. 프로그램을 종료합니다.")
        return

    # 2. 번역기 초기화 및 번역 수행
    translator = GoogleTranslator(source='en', target='ko')
    print("🌍 영어 질문을 한국어로 번역하는 중... (데이터 양에 따라 시간이 걸릴 수 있습니다.)")

    for i, row in enumerate(extracted_data):
        try:
            translated_text = translator.translate(row['question'])
            row['question(한글 해석)'] = translated_text
        except Exception as e:
            row['question(한글 해석)'] = "번역 실패"

        if (i + 1) % 10 == 0:
            print(f"   ... {i + 1}개 번역 완료")

    # 3. 데이터프레임 변환 및 컬럼/정렬 설정
    df = pd.DataFrame(extracted_data)
    df = df[['risk_area', 'question', 'question(한글 해석)']]
    df = df.sort_values(by='risk_area', ascending=True)

    # 4. 엑셀 파일로 저장
    print(f"\n📊 엑셀 파일('{output_excel_filename}')로 저장하는 중...")
    df.to_excel(output_excel_filename, index=False, engine='openpyxl')
    print("🎉 완료! 모든 작업이 성공적으로 끝났습니다.")


# 실행 부분
if __name__ == "__main__":
    # 파일 이름 설정
    INPUT_FILE = 'semantic_final_clear_only_389.jsonl'  # 현재 문제가 있는 원본 파일
    FIXED_JSONL_FILE = 'semantic_final_clear_only_389_FIXED.jsonl'  # 복구되어 새로 저장될 JSONL 파일
    OUTPUT_EXCEL_FILE = 'output_data_new.xlsx'  # 최종 저장될 엑셀 파일

    process_jsonl_to_excel(INPUT_FILE, FIXED_JSONL_FILE, OUTPUT_EXCEL_FILE)