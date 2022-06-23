from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle


tokenizer = Tokenizer()
max_len = 271
loaded_model = load_model('best_model.h5')

with open ('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

tokenizer2 = Tokenizer()
max_len2 = 271
loaded_model2 = load_model('best_model_se.h5')

with open ('tokenizer.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)

def spam_predict(content):
    tokenizer.fit_on_texts(content)
    print(content)
    content_encoded = tokenizer.texts_to_sequences([content])   #받은 사용자 텍스트를 정수 인코딩
    print(content_encoded)
    content_pad_new = pad_sequences(content_encoded, maxlen = max_len) #받은 사용자 텍스트를 패딩
    predict_content = float(loaded_model.predict(content_pad_new)) #예측
    
    print("검증 데이터의 정확도는 %.4f 입니다. \n" % (predict_content))
    if(predict_content > 0.5):
        print("{:.2f}% 확률로 스미싱 문자입니다. \n".format(predict_content *100))
        return 1
    else:
        print("{:.2f}% 확률로 정상 메세지입니다.\n".format((1-predict_content)*100))
        return 0

#spam_predict("칭찬이야. 간만에 남 칭찬했더니 막 식은땀이 나는데.")
#spam_predict("[국제발신] 김00님 [874,025]원 해외승인 본인 구매결제 아닐시 신고요망 Paypal 고객센터 : 052-5058-5464")

def spam_predict_se(content):
    tokenizer2.fit_on_texts(content)
    print(content)
    content_encoded = tokenizer2.texts_to_sequences([content])  # 받은 사용자 텍스트를 정수 인코딩
    print(content_encoded)
    content_pad_new = pad_sequences(content_encoded, maxlen=max_len2)  # 받은 사용자 텍스트를 패딩
    predict_content = float(loaded_model2.predict(content_pad_new))  # 예측

    print("검증 데이터의 정확도는 %.4f 입니다. \n" % (predict_content))
    if (predict_content > 0.5):
        print("{:.2f}% 확률로 스미싱 전화번호 입니다. \n".format(predict_content * 100))
        return 1
    else:
        print("{:.2f}% 확률로 정상 전화번호 입니다.\n".format((1 - predict_content) * 100))
        return 0
