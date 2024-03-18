from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import scipy as sp

# 샘플 모델과 타겟 문장
model_couple = ["좋아해", "사랑해", "너 닮았다", "딱 자긴데", "너무 좋아", "보고 싶어"]
model_some = ["그럴 수 있지", "괜찮아", "언제 볼래", "뭐 할래", "시간 괜찮아", "같이 먹으러 갈래"]
model_business = ["알겠습니다", "확인했습니다", "감사합니다", "그때 뵙겠습니다", "드립니다", "바랍니다"]
model_friend = ["야", "아니", "너", "ㅇㅇ", "ㅇㅋ", "ㄴㄴ", "ㅇㅈ", "ㄷㄷ", "개", "바보"]

models = [model_couple, model_some, model_business, model_friend]
target = ["나 너 정말 좋아해, 하늘만큼 땅만큼. 평생 너만 사랑하고 싶어. 보고 싶어."]

# TfidfVectorizer 초기화, 형태소 분석기 초기화(Okt 사용)
vectorizer = TfidfVectorizer(min_df=1)
translate = Okt()

# 모든 문장에 대해 형태소 분석, 분석 결과를 공백으로 구분된 문자열로 변환
models_tokens = [[translate.morphs(sentence) for sentence in model] for model in models]
models_for_vectorize = [' '.join(tokens) for tokens_list in models_tokens for tokens in tokens_list]

X2 = vectorizer.fit_transform(models_for_vectorize)  # TF-IDF 행렬 생성

# 대상 문장에 대해 형태소 분석, 분석 결과를 공백으로 구분된 문자열로 변환
target_tokens = [translate.morphs(row) for row in target]
target_for_vectorize = [' '.join(tokens) for tokens in target_tokens]

t_vec = vectorizer.transform(target_for_vectorize)  # 대상 문장의 TF-IDF 벡터 생성


# 두 벡터 간의 거리 계산
def dist_raw(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError("두 벡터의 크기가 다릅니다.")
    delta = v1 - v2
    delta_arr = delta.toarray()  # TF-IDF 벡터를 배열로 변환
    return sp.linalg.norm(delta_arr)


# 각 모델과 대상 문장 간의 평균 거리 계산
average_distances = []  # 모델 그룹별 평균 거리를 저장할 리스트

for group_idx, models_group in enumerate(models_tokens):
    group_distances = []  # 모델 그룹 내의 모든 거리를 저장할 리스트
    for i, model_sentence in enumerate(models_group):
        d = dist_raw(X2[i], t_vec[0])  # 대상 문장의 TF-IDF 벡터와의 거리 계산
        group_distances.append(d)

    # 모델 그룹의 평균 거리 계산
    average_distance = sum(group_distances) / len(group_distances)
    average_distances.append(average_distance)

    print(f"=== 모델 그룹 {group_idx+1} ===")
    print("평균 거리:", average_distance)
    print()

# 평균 거리가 가장 작은 모델 그룹 찾기
best_group_idx = average_distances.index(min(average_distances))
best_average_distance = min(average_distances)

print("== Best Group:", best_group_idx)
print("== Best Average Distance:", best_average_distance)
