import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import numpy as np

# 대화 내용 모델 그룹
conversations = {
    "썸": ["좋아해", "사랑해", "너 닮았다", "딱 자긴데", "너무 좋아", "보고 싶어"],
    "연애": ["좋아해", "사랑해"],
    "비즈니스": ["알겠습니다", "확인했습니다", "감사합니다", "그때 뵙겠습니다", "드립니다", "바랍니다"],
    "친구": ["야", "아니", "너", "ㅇㅇ", "ㅇㅋ", "ㄴㄴ", "ㅇㅈ", "ㄷㄷ", "개", "바보"]
}

# 대화 내용 모델 그룹을 하나의 문서로 합침
combined_conversations = {relation: ' '.join(conversation) for relation, conversation in conversations.items()}

# 형태소 분석기 초기화(Okt 사용)
translate = Okt()

# 대화 내용 모델 그룹을 하나의 문서로 합친 것도 형태소 분석
combined_tokens = {relation: translate.morphs(text) for relation, text in combined_conversations.items()}
combined_for_vectorize = {relation: ' '.join(tokens) for relation, tokens in combined_tokens.items()}

# TF-IDF 벡터 생성
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(list(combined_for_vectorize.values()))


# 두 벡터 간의 거리 계산 (유클리드 거리)
def dist_euclidean(v1, v2):
    delta = v1 - v2
    return np.sqrt(np.sum(delta.power(2)))


# 대화 내용과 관계 유형 간의 유사도 계산
def classify_relationship(target_input):
    # 대상 문장에 대해 형태소 분석, 분석 결과를 공백으로 구분된 문자열로 변환
    target_tokens = translate.morphs(target_input)
    target_for_vectorize = ' '.join(target_tokens)
    t_vec = vectorizer.transform([target_for_vectorize])  # 대상 문장의 TF-IDF 벡터 생성

    # 각 관계 유형과 대상 문장 간의 코사인 유사도 계산
    cosine_similarities = {}
    for relation, combined_vector in combined_for_vectorize.items():
        combined_vec = vectorizer.transform([combined_vector])
        cosine_similarities[relation] = cosine_similarity(t_vec, combined_vec)[0][0]

    # 가장 높은 코사인 유사도를 가진 관계 유형 반환
    closest_relation_cosine = max(cosine_similarities, key=cosine_similarities.get)

    # 각 관계 유형과 대상 문장 간의 유클리드 거리 계산
    euclidean_distances = {}
    for relation, combined_vector in combined_for_vectorize.items():
        combined_vec = vectorizer.transform([combined_vector]).toarray()[0]
        target_vec_arr = t_vec.toarray()[0]
        euclidean_distances[relation] = euclidean(combined_vec, target_vec_arr)

    # 가장 짧은 유클리드 거리를 가진 관계 유형 반환
    closest_relation_euclidean = min(euclidean_distances, key=euclidean_distances.get)

    return closest_relation_cosine, closest_relation_euclidean


# CSV 파일에서 특정 범위의 대화 내용을 읽어오는 함수
def read_csv_file(filename, start_row, end_row, column_index):
    conversation = ""
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        for i, row in enumerate(reader):
            if i < start_row:
                continue
            if i > end_row:
                break
            conversation += row[column_index] + " "  # 지정된 열의 대화를 하나의 문자열로 합침
    return conversation


# 사용자로부터 입력 받기
filename = "conversation_data2.csv"
start_row = int(input("시작 행을 입력하세요: "))
end_row = int(input("끝 행을 입력하세요: "))
column_index = 2    # int(input("분석할 열 인덱스를 입력하세요: "))

conversations = read_csv_file(filename, start_row, end_row, column_index)

# 대화 내용과 관계 유형 판단
relation_cosine, relation_euclidean = classify_relationship(conversations)
print("코사인 유사도를 기반으로 한 결과:", relation_cosine)
print("유클리드 거리를 기반으로 한 결과:", relation_euclidean)
