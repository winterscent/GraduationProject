import csv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import numpy as np
import os
import re

# 대화 내용 모델 그룹
conversations = {
    "썸": ["이모티콘", "좋아", "좋아해", "웅", "보고 싶어"],
    "연애": ["이모티콘", "좋아", "좋아해", "사랑", "사랑해", "웅", "보고 싶어"],
    "친구": ["야", "아니", "너", "ㅇㅇ", "ㅇㅋ", "ㄴㄴ", "ㅇㅈ", "ㄷㄷ", "개", "꺼져"],
    "비즈니스": ["알겠습니다", "확인했습니다", "감사합니다", "그때 뵙겠습니다", "드립니다", "바랍니다", "네"]
}

# 대화 내용 모델 그룹을 하나의 문서로 합침
combined_conversations = {relation: ' '.join(conversation) for relation, conversation in conversations.items()}

# 형태소 분석기 초기화(Okt 사용)
translate = Okt()

# 대화 내용 모델 그룹을 형태소 분석
combined_tokens = {relation: translate.morphs(text) for relation, text in combined_conversations.items()}
combined_for_vectorize = {relation: ' '.join(tokens) for relation, tokens in combined_tokens.items()}

# TF-IDF 벡터 생성
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(list(combined_for_vectorize.values()))


# 대화 내용과 관계 유형 간의 유사도 계산
def classify_relationship(target_input):
    target_tokens = translate.morphs(target_input)
    target_for_vectorize = ' '.join(target_tokens)
    t_vec = vectorizer.transform([target_for_vectorize])  # 대상 문장의 TF-IDF 벡터 생성

    # 코사인 유사도 계산
    cosine_similarities = {}
    for relation, combined_vector in combined_for_vectorize.items():
        combined_vec = vectorizer.transform([combined_vector])
        cosine_similarities[relation] = cosine_similarity(t_vec, combined_vec)[0][0]

    # 유클리드 거리 계산
    euclidean_distances = {}
    for relation, combined_vector in combined_for_vectorize.items():
        combined_vec = vectorizer.transform([combined_vector]).toarray()[0]
        target_vec_arr = t_vec.toarray()[0]
        euclidean_distances[relation] = euclidean(combined_vec, target_vec_arr)

    # 최종 점수 계산 (코사인 유사도는 최대화, 유클리드 거리는 최소화가 목적)
    final_scores = {}
    max_euclidean = np.max(list(euclidean_distances.values()))
    for relation in cosine_similarities.keys():
        cosine_score = cosine_similarities[relation]
        euclidean_score = euclidean_distances[relation]
        adjusted_euclidean_score = 1 - (euclidean_score / max_euclidean)
        final_score = (cosine_score + adjusted_euclidean_score) / 2
        final_scores[relation] = final_score

    closest_relation_final = max(final_scores, key=final_scores.get)

    return cosine_similarities, euclidean_distances, final_scores, closest_relation_final


def read_csv_file_by_date(filename, start_date, end_date, column_index):
    conversation = ""
    first_date = None
    last_date = None
    date_format = "%Y-%m-%d %H:%M:%S"  # CSV 파일의 날짜 형식 정의

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵

        for row in reader:
            try:
                row_date = datetime.strptime(row[0], date_format).strftime("%Y%m%d")  # 날짜 형식을 YYYYMMDD로 통일
                if not first_date:
                    first_date = row_date  # 첫 번째 날짜 저장
                last_date = row_date  # 마지막 날짜 업데이트
            except ValueError:
                print("잘못된 날짜 형식이 포함된 행을 건너뜁니다.")
                continue
            if start_date <= row_date <= end_date:
                conversation += row[column_index] + " "

    return conversation, first_date, last_date


def read_txt_file_by_date(filename, start_date, end_date):
    conversation = ""
    first_date = None
    last_date = None

    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

        # 날짜 및 시간 패턴을 추출하기 위한 정규식
        pattern = r"(\d{4}\. \d{1,2}\. \d{1,2}\. .*\d{1,2}:\d{2}), (.*?): (.*)"
        matches = re.findall(pattern, content)

        for match in matches:
            date_str, user, message = match
            try:
                row_date = datetime.strptime(date_str, "%Y. %m. %d. %p %I:%M").strftime("%Y%m%d")  # 예: "2022. 6. 8. 오후 7:46"
                if not first_date:
                    first_date = row_date  # 첫 번째 날짜 저장
                last_date = row_date  # 마지막 날짜 업데이트
            except ValueError:
                print("잘못된 날짜 형식을 건너뜁니다.")
                continue
            if start_date <= row_date <= end_date:
                conversation += message + " "

    return conversation, first_date, last_date


def validate_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y%m%d")
    except ValueError:
        return None


def get_valid_file_path():
    while True:
        filename = input("CSV 또는 TXT 파일 경로를 입력하세요 (예: filename.csv 또는 filename.txt): ")
        if os.path.exists(filename) and (filename.endswith('.csv') or filename.endswith('.txt')):
            return filename
        else:
            print("파일이 없거나 잘못된 형식입니다. 다시 입력해주세요.")


# 파일 경로 입력받기
filename = get_valid_file_path()

# 파일 형식에 따라 첫 번째와 마지막 날짜 추출 및 대화 내용 추출
if filename.endswith('.csv'):
    column_index = 2  # 분석할 열의 인덱스 (대화 내용이 있는 열)
    start_date_str = input("시작 날짜를 입력하세요 (예: 20240101): ")
    start_date = validate_date(start_date_str)
    while start_date is None:
        print("정확한 날짜 형식으로 기입해주세요.")
        start_date_str = input("시작 날짜를 입력하세요 (예: 20240101): ")
        start_date = validate_date(start_date_str)

    end_date_str = input("끝 날짜를 입력하세요 (예: 20241231): ")
    end_date = validate_date(end_date_str)
    while end_date is None:
        print("정확한 날짜 형식으로 기입해주세요.")
        end_date_str = input("끝 날짜를 입력하세요 (예: 20241231): ")
        end_date = validate_date(end_date_str)

    # 대화 내용을 읽어오기
    conversations, first_date, last_date = read_csv_file_by_date(filename, start_date, end_date, column_index)

else:  # TXT 파일 처리
    start_date_str = input("시작 날짜를 입력하세요 (예: 20240101): ")
    start_date = validate_date(start_date_str)
    while start_date is None:
        print("정확한 날짜 형식으로 기입해주세요.")
        start_date_str = input("시작 날짜를 입력하세요 (예: 20240101): ")
        start_date = validate_date(start_date_str)

    end_date_str = input("끝 날짜를 입력하세요 (예: 20241231): ")
    end_date = validate_date(end_date_str)
    while end_date is None:
        print("정확한 날짜 형식으로 기입해주세요.")
        end_date_str = input("끝 날짜를 입력하세요 (예: 20241231): ")
        end_date = validate_date(end_date_str)

    # 대화 내용을 읽어오기
    conversations, first_date, last_date = read_txt_file_by_date(filename, start_date, end_date)

# 파일에 포함된 날짜 출력
print(f"파일에 포함된 날짜: {first_date} ~ {last_date}")

# 대화 내용과 관계 유형 판단
cosine_similarities, euclidean_distances, final_scores, closest_relation_final = classify_relationship(conversations)

print("코사인 유사도 기반 점수(1에 가까울수록 좋음):")
for relation, score in cosine_similarities.items():
    print("{}: {:.4f}".format(relation, score))

print("\n유클리드 거리 기반 점수(0에 가까울수록 좋음):")
for relation, score in euclidean_distances.items():
    print("{}: {:.4f}".format(relation, score))

print("\n최종 점수(1에 가까울수록 좋음):")
for relation, score in final_scores.items():
    print("{}: {:.4f}".format(relation, score))

print("\n입력한 대화 내용은 '{}' 관계에 해당합니다.".format(closest_relation_final))
