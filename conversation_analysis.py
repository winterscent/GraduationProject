import csv
from datetime import datetime
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

# 딥러닝 모델과 벡터화기, 라벨 사전 로드
model = load_model('relationship_model.h5')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_dict = joblib.load('label_dict.pkl')
inverse_label_dict = {v: k for k, v in label_dict.items()}

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

# 대화 내용 모델 그룹을 하나의 문서로 합친 것도 형태소 분석
combined_tokens = {relation: translate.morphs(text) for relation, text in combined_conversations.items()}
combined_for_vectorize = {relation: ' '.join(tokens) for relation, tokens in combined_tokens.items()}

# TF-IDF 벡터 생성
X = vectorizer.fit_transform(list(combined_for_vectorize.values()))


# 대화 내용과 관계 유형 간의 유사도 계산
def classify_relationship(target_input):
    # 대상 문장에 대해 형태소 분석, 분석 결과를 공백으로 구분된 문자열로 변환
    target_tokens = translate.morphs(target_input)
    target_for_vectorize = ' '.join(target_tokens)
    t_vec = vectorizer.transform([target_for_vectorize])  # 대상 문장의 TF-IDF 벡터 생성

    # 딥러닝 모델 예측
    prediction = model.predict(t_vec.toarray())
    predicted_label = inverse_label_dict[np.argmax(prediction)]

    # 각 관계 유형과 대상 문장 간의 코사인 유사도 계산
    cosine_similarities = {}
    for relation, combined_vector in combined_for_vectorize.items():
        combined_vec = vectorizer.transform([combined_vector])
        cosine_similarities[relation] = cosine_similarity(t_vec, combined_vec)[0][0]

    # 각 관계 유형과 대상 문장 간의 유클리드 거리 계산
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
        # 유클리드 거리를 최대화된 유클리드 거리로 나누어 점수화
        adjusted_euclidean_score = 1 - (euclidean_score / max_euclidean)
        final_score = (cosine_score + adjusted_euclidean_score) / 2
        final_scores[relation] = final_score

    closest_relation_final = max(final_scores, key=final_scores.get)

    return cosine_similarities, euclidean_distances, final_scores, closest_relation_final, predicted_label


def read_csv_file_by_date(filename, start_date, end_date, column_index):
    conversation = ""
    date_format = "%Y-%m-%d %H:%M:%S"  # CSV 파일의 날짜 형식 정의
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # 헤더 스킵

        for row in reader:
            try:
                row_date = datetime.strptime(row[0], date_format)
            except ValueError:
                print("잘못된 날짜 형식이 포함된 행을 건너뜁니다.")
                continue
            if start_date <= row_date <= end_date:
                conversation += row[column_index] + " "
    return conversation


def validate_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def get_valid_file_path():
    while True:
        filename = input("CSV 파일 경로를 입력하세요 (예: filename.csv): ")
        if os.path.exists(filename) and filename.endswith('.csv'):
            return filename
        else:
            print("파일이 없거나 잘못된 형식입니다. 다시 입력해주세요.")


# 파일 경로 입력받기
filename = get_valid_file_path()

# 1열의 날짜를 읽어 사용자에게 보여주기
dates = []
date_format = "%Y-%m-%d %H:%M:%S"  # CSV 파일의 날짜 형식 정의
with open(filename, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 스킵
    for row in reader:
        dates.append(row[0])

print("파일에 포함된 날짜:", ", ".join(dates[:1]), "~", ", ".join(dates[-1:]))  # 처음 1개와 마지막 1개의 날짜를 출력

# 시작 날짜 입력 받기
start_date_str = input("시작 날짜를 입력하세요 (예: 2024-01-01): ")
start_date = validate_date(start_date_str)
while start_date is None:
    print("정확한 날짜 형식으로 기입해주세요.")
    start_date_str = input("시작 날짜를 입력하세요 (예: 2024-01-01): ")
    start_date = validate_date(start_date_str)

# 끝 날짜 입력 받기
end_date_str = input("끝 날짜를 입력하세요 (예: 2024-12-31): ")
end_date = validate_date(end_date_str)
while end_date is None:
    print("정확한 날짜 형식으로 기입해주세요.")
    end_date_str = input("끝 날짜를 입력하세요 (예: 2024-12-31): ")
    end_date = validate_date(end_date_str)

column_index = 2  # 분석할 열의 인덱스 (대화 내용이 있는 열)

# 선택한 날짜 범위 내의 대화 내용을 읽어오기
conversations = read_csv_file_by_date(filename, start_date, end_date, column_index)

# 대화 내용과 관계 유형 판단
cosine_similarities, euclidean_distances, final_scores, closest_relation_final, predicted_label = classify_relationship(conversations)

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
print("\n딥러닝 모델에 의해 예측된 관계는 '{}' 입니다.".format(predicted_label))
