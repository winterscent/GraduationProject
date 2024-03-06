from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import scipy as sp

# 샘플 모델과 타겟 문장
models = ['좋아해', '사랑해', '너 닮았다', '딱 자긴데', '너무 좋아', '보고 싶어']
target = ['나 너 정말 좋아해, 하늘만큼 땅만큼. 평생 너만 사랑하고 싶어.']

# TfidfVectorizer 초기화, 형태소 분석기 초기화(Okt 사용)
vectorizer = TfidfVectorizer(min_df = 1)
translate = Okt()

# 모든 문장에 대해 형태소 분석, 분석 결과를 공백으로 구분된 문자열로 변환
models_tokens = [translate.morphs(row) for row in models]
models_for_vectorize = [' '.join(tokens) for tokens in models_tokens]

X2 = vectorizer.fit_transform(models_for_vectorize)     # TF-IDF 행렬 생성

# 각 문장에 대해 형태소 분석, 분석 결과를 공백으로 구분된 문자열로 변환
target_tokens = [translate.morphs(row) for row in target]
targets_for_vectorize = [' '.join(tokens) for tokens in target_tokens]

t_vec = vectorizer.transform(targets_for_vectorize)     # TF-IDF 행렬 생성

# 두 벡터 간의 거리 계산
def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

best_dist = 65535       # 초기값으로 매우 큰 값 설정
best_i = None       # 가장 가까운 문장의 인덱스를 저장할 변수
s = []      # 거리와 인덱스를 저장할 리스트

# 각 모델과 대상 문장 간의 거리 계산
for i in range(len(models)):
    vec = X2.getrow(i)      # i번째 모델의 TF-IDF 벡터 가져오기
    d = dist_raw(vec, t_vec)        # 두 벡터 간의 거리 계산
    print("== Post %i with dist=%.2f : %s"%(i, d, models[i]))

    # 현재까지의 최소 거리보다 작은 거리를 가진 경우 계산
    if d < best_dist:
        best_dist, best_i = d, i
        
    s.append([d, i])
print("== Best %i with dist=%.6f : %s"%(best_i, best_dist, models[best_i]))