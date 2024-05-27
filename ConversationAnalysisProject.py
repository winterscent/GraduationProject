import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import numpy as np

# 대화 내용 모델 그룹
conversations = {
    "썸": ["좋아", "좋아해", "웅", "보고 싶어"],
    "연애": ["좋아", "좋아해", "사랑", "사랑해", "웅", "보고 싶어"],
    "비즈니스": ["알겠습니다", "확인했습니다", "감사합니다", "그때 뵙겠습니다", "드립니다", "바랍니다", "네"],
    "친구": ["야", "아니", "너", "ㅇㅇ", "ㅇㅋ", "ㄴㄴ", "ㅇㅈ", "ㄷㄷ", "개", ""]
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

    return (cosine_similarities, euclidean_distances, final_scores, closest_relation_final)


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
filename = "conversation_data3.csv"
start_row = int(input("시작 행을 입력하세요: "))
end_row = int(input("끝 행을 입력하세요: "))
column_index = 2

conversations = read_csv_file(filename, start_row, end_row, column_index)

# 대화 내용과 관계 유형 판단
cosine_similarities, euclidean_distances, final_scores, closest_relation_final = classify_relationship(conversations)

print("코사인 유사도 기반 점수:")
for relation, score in cosine_similarities.items():
    print("{}: {:.4f}".format(relation, score))

print("\n유클리드 거리 기반 점수:")
for relation, score in euclidean_distances.items():
    print("{}: {:.4f}".format(relation, score))

print("\n최종 점수:")
for relation, score in final_scores.items():
    print("{}: {:.4f}".format(relation, score))

print("\n입력한 대화 내용은 '{}' 관계에 해당합니다.".format(closest_relation_final))
