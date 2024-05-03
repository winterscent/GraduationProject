from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import scipy as sp

# 대화 내용 모델 그룹
conversations = {
    "썸": ["좋아해", "사랑해", "너 닮았다", "딱 자긴데", "너무 좋아", "보고 싶어"],
    "연애": ["좋아해", "사랑해"],
    "비즈니스": ["알겠습니다", "확인했습니다", "감사합니다", "그때 뵙겠습니다", "드립니다", "바랍니다"],
    "친구": ["야", "아니", "너", "ㅇㅇ", "ㅇㅋ", "ㄴㄴ", "ㅇㅈ", "ㄷㄷ", "개", "바보"]
}

# 대화 내용 모델 그룹을 하나의 문서로 합침
combined_conversations = {relation: ' '.join(conversation) for relation, conversation in conversations.items()}

# TfidfVectorizer 초기화, 형태소 분석기 초기화(Okt 사용)
vectorizer = TfidfVectorizer(min_df=1)
translate = Okt()

# 대화 내용 모델 그룹을 하나의 문서로 합친 것도 형태소 분석
combined_tokens = {relation: translate.morphs(row) for relation, row in combined_conversations.items()}
combined_for_vectorize = {relation: ' '.join(tokens) for relation, tokens in combined_tokens.items()}

X2 = vectorizer.fit_transform(list(combined_for_vectorize.values()))  # TF-IDF 행렬 생성


# 두 벡터 간의 거리 계산
def dist_raw(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError("두 벡터의 크기가 다릅니다.")
    delta = v1 - v2
    delta_arr = delta.toarray()  # TF-IDF 벡터를 배열로 변환
    return sp.linalg.norm(delta_arr)


# 대화 내용과 관계 유형 간의 거리 계산
def classify_relationship(target_input):
    # 대상 문장에 대해 형태소 분석, 분석 결과를 공백으로 구분된 문자열로 변환
    target_tokens = translate.morphs(target_input)
    target_for_vectorize = ' '.join(target_tokens)
    t_vec = vectorizer.transform([target_for_vectorize])  # 대상 문장의 TF-IDF 벡터 생성

    # 각 관계 유형과 대상 문장 간의 거리 계산
    distances = {}
    for relation, combined_vector in combined_for_vectorize.items():
        d = dist_raw(X2[list(combined_for_vectorize.keys()).index(relation)], t_vec[0])  # 대상 문장의 TF-IDF 벡터와의 거리 계산
        distances[relation] = d

    # 가장 거리가 짧은 관계 유형 반환
    closest_relation = min(distances, key=distances.get)
    return closest_relation


# 사용자로부터 대화 내용 입력 받기
target_input = input("대화 내용을 입력하세요: ")

# 대화 내용과 관계 유형 판단
relation = classify_relationship(target_input)
print("입력한 대화 내용은 '{}' 관계에 해당합니다.".format(relation))
