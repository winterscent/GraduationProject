import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import numpy as np

# CSV 파일에서 데이터 로드
data = pd.read_csv("labeled_data_for_modeltraining.csv")  # 라벨링된 대화 내용 CSV 파일
texts = data["text"].tolist()
labels = data["label"].tolist()

# 형태소 분석기 초기화
okt = Okt()


# 텍스트 전처리 및 토큰화
def preprocess_text(text):
    tokens = okt.morphs(text)
    return " ".join(tokens)


texts = [preprocess_text(text) for text in texts]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(min_df=1, max_features=5000)
X = vectorizer.fit_transform(texts).toarray()

# 라벨 인코딩
label_dict = {"썸": 0, "연애": 1, "친구": 2, "비즈니스": 3}
y = np.array([label_dict[label] for label in labels])

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 정의
model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4, activation="softmax"))  # 클래스 수에 맞게 출력 뉴런 수를 4로 설정

# 모델 컴파일
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# 모델 학습
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {accuracy * 100:.2f}%")

# 모델 저장
model.save("relationship_model.h5")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_dict, "label_dict.pkl")
