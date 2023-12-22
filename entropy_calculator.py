import collections
import math

def calculate_entropy(file_path):
    # 파일 읽기
    with open(file_path, 'rb') as file:
        data = file.read()

    # 각 바이트 값의 발생 빈도를 계산
    byte_counters = collections.Counter(data)
    file_length = len(data)

    # 샤논 엔트로피 계산
    entropy = 0
    for count in byte_counters.values():
        # 각 바이트의 확률
        probability = count / file_length
        # 엔트로피 계산
        entropy -= probability * math.log2(probability)

    return entropy
