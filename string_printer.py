import pefile
import re

def get_ascii_strings(raw_data, min_length):
    pattern = b'[\x20-\x7E]{' + str(min_length).encode() + b',}'
    return re.findall(pattern, raw_data)

def get_unicode_strings(raw_data, min_length):
    pattern = b'(?:[\x20-\x7E][\x00]){' + str(min_length).encode() + b',}'
    return re.findall(pattern, raw_data)

def is_valid_string(s):  
    # 바이트 타입인 경우, 문자열로 디코딩
    if isinstance(s, bytes):
        s = s.decode()

    #문자의 길이가 3자 이하인 경우 필터링
    if len(s) <= 3:
        return False

    # 정규 표현식을 사용하여 기본 패턴 검사
    if not re.match(r'^[a-zA-Z0-9\s.,;:\'\"!?()-]+$', s):
        return False

    # 영문자와 숫자가 아닌 문자가 일정 비율 이상인 경우 필터링
    non_alnum_chars = sum(not c.isalnum() for c in s)
    if non_alnum_chars / len(s) > 0.3:  # 예를 들어, 비문자/숫자가 30% 이상인 경우
        return False

    # 반복되는 문자가 일정 비율 이상인 경우 필터링 (예: 'wwwwwwwwwx')
    if len(set(s)) / len(s) < 0.4: 
        return False

    return True

def print_strings(file_path, min_l=4):
    try:
        pe = pefile.PE(file_path)

        all_ascii_strings = []
        all_unicode_strings = []

        print("\nStrings Analysis:")
        for section in pe.sections:
            raw_data = section.get_data()
            ascii_strings = get_ascii_strings(raw_data, min_length=min_l)
            unicode_strings = get_unicode_strings(raw_data, min_length=min_l)

            section_name = section.Name.decode().rstrip('\x00')
            print(f"\n Section: {section_name}")

            if ascii_strings:
                print("  ASCII Strings:")
                for string in ascii_strings:
                    decoded_string = string.decode()
                    if is_valid_string(decoded_string):
                        all_ascii_strings.append(decoded_string)
                        print(f"  {decoded_string}")

            if unicode_strings:
                print("\n Unicode Strings:")
                for string in unicode_strings:
                    all_unicode_strings.append(string.decode('utf-16'))
                    print(f"  {string}")

        return all_ascii_strings, all_unicode_strings

    except pefile.PEFormatError as e:
        print(f"Error: {e}")
        return [], []

