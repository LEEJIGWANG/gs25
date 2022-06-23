import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import pickle
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

###Pretreatment

data = pd.read_excel('C:/Users/user/Desktop/FlaskTest/s_data.xlsx')
print(data)  # 데이터 불러오기

print("총 샘플의 수 : \n ", data.count())  # 샘플 수
data.drop_duplicates(subset=['메세지 내용'], inplace=True)
print('총 샘플의 수 :', len(data))  # 중복 값 제외

print('정상 메일과 스팸 메일의 개수')
print(data.groupby('스팸여부').size().reset_index(name='count'))
print('\n')
print(f'정상 메일의 비율 = {round(data["스팸여부"].value_counts()[0] / len(data) * 100, 3)}%')
print(f'스팸 메일의 비율 = {round(data["스팸여부"].value_counts()[1] / len(data) * 100, 3)}%')

X_data = data['메세지 내용']
y_data = data['스팸여부']
print('메일 본문의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

print('--------훈련 데이터의 비율-----------')
print(f'정상 메일 = {round(y_train.value_counts()[0] / len(y_train) * 100, 3)}%')
print(f'스팸 메일 = {round(y_train.value_counts()[1] / len(y_train) * 100, 3)}%')

print('--------테스트 데이터의 비율-----------')
print(f'정상 메일 = {round(y_test.value_counts()[0] / len(y_test) * 100, 3)}%')
print(f'스팸 메일 = {round(y_test.value_counts()[1] / len(y_test) * 100, 3)}%')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_encoded = tokenizer.texts_to_sequences(X_train)
print(X_train_encoded[:5])

word_to_index = tokenizer.word_index
print(word_to_index)

threshold = 2
total_cnt = len(word_to_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if (value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

vocab_size = len(word_to_index) + 1
print('단어 집합의 크기: {}'.format((vocab_size)))
print('메시지의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded))
print('메세지의 평균 길이 : %f' % (sum(map(len, X_train_encoded)) / len(X_train_encoded)))
plt.hist([len(sample) for sample in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')

max_len = 271
X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len)
print("훈련 데이터의 크기(shape):", X_train_padded.shape)

###RNN


embedding_dim = 32
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model_se.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train_padded, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len)
"""print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test_padded, y_test)[1]))"""

loaded_model = load_model('best_model_se.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test_padded, y_test)[1]))

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)


def spam_predict(content):
    print(content)
    content_encoded = tokenizer.texts_to_sequences([content])  # 받은 사용자 텍스트를 정수 인코딩
    print(content_encoded)
    content_pad_new = pad_sequences(content_encoded, maxlen=max_len)  # 받은 사용자 텍스트를 패딩
    predict_content = float(loaded_model.predict(content_pad_new))  # 예측

    print("검증 데이터의 정확도는 %.4f 입니다. \n" % (predict_content))
    if (predict_content > 0.5):
        print("{:.2f}% 확률로 스미싱 전화번호 입니다. \n".format(predict_content * 100))
    else:
        print("{:.2f}% 확률로 정상 전화번호 입니다.\n".format((1 - predict_content) * 100))

# 정상 - spam_predict("[국제발신] 김00님 [874,025]원 해외승인 본인 구매결제 아닐시 신고요망 Paypal 고객센터 : 052-5058-5464"
# 정상 - spam_predict("애 엄마가 말 안해요? 나 사업 자금 밀어준다든데?")
# 정상 - spam_predict("요금을 청구할 수 없습니다. 계정에 금액을 충전한 후 넷플릭스 앱에서 결제를 다시 시도해주세요")

