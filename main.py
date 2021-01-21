import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re

# Dataset'imiz
data = pd.read_csv('dataset.csv')

# Dataset'in sutunları (tweet_id = tweet'in idsi, sentiment = duygular, author = yazar ismi, content = tweet)
data.info()

# Dataset'ten "tweet_id" ve "author" stunları çıkarıyoruz.
data = data.drop(columns=['tweet_id', 'author'])

# Dataset'imizi tekrardan "sentiment" ve "content" stunları olacka şekilde belirtiyoruz.
data = data[['sentiment', 'content']]
data.info()

# Dataset'imizde bulunan 13 adet duyguyu haritalandırıyoruz.
data['sentiment'] = data['sentiment'].map({'neutral': 0, 'worry': 1, 'happiness': 2,
                                           'sadness': 3,
                                           'love': 4,
                                           'surprise': 5,
                                           'fun': 6,
                                           'relief': 7,
                                           'hate': 8,
                                           'empty': 9,
                                           'enthusiasm': 10,
                                           'boredom': 11,
                                           'anger': 12})

# tweet içeriklerimizi daha doğru sonuç alabilmek için ilk önce tüm harfleri küçük yapıp sonra gereksiz işaretleri çıkartıyoruz.
data['content'] = data['content'].apply(lambda x: x.lower())
data['content'] = data['content'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

# Kelimelerin LSTM ağında işlenebilmesi için tokenizer kullanıyoruz. Vektörleştirirken özellik sayısını 5000 olarak tanımladım.
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(data['content'].values)
dataToken = tokenizer.texts_to_sequences(data['content'].values)
dataToken = pad_sequences(dataToken)

# LSTM ağının oluşturuyoruz.  activation='softmax', softmax aktivasyon modelini duygu analizinde en uygun gördüm.
output_dimen = 128
units = 196
model = Sequential()
model.add(Embedding(5000, output_dimen, input_length=dataToken.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(units, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
# Modelimizin eğitimi için kullanılan parametreler
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(dataToken, Y, test_size=0.33, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 32 partilik bir değer seçerek değer/tahmin değerini olabildiğince yükset tutuyorum (işlemcimin el verdiği kadar)
batch_length = 32
# epochs minimum tavsiye dilen en az değeri 7. Her epoch değeri ortalama 60-70 saniye arasında sürüyor.
# Eğitimin örnekleme değerini 7 olarak tuttum. Daha yüksek değerde daha iyi sonuçlar alınmaktadır.
model.fit(X_train, Y_train, epochs=7, batch_size=batch_length, verbose=1)



validationSize = 2000
X_val = X_test[-validationSize:]
Y_val = Y_test[-validationSize:]
X_test = X_test[:-validationSize]
Y_test = Y_test[:-validationSize]
score, average = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_length)
# Eğitim setimizin emojiler arası skor dağılımı ve averajı. 1-6 epoch değeri arasında yakın değerler alınırken 7 ve üstünde tahmin değeri artmaktadır.
print("Rate: %.2f" % (score))
print("AVG: %.2f" % (average))

# -------------------------------------Başlangıç---------------------------------------------------------

# Eğitim verilerinde pozitif ve negatif çıkarımların dağılımı ölçülüyor. Çıkarımlar arası fazla olması biraz olumsuz oldu.
# Bu değerleri yaklaştırmak için bir kaç benzer ve eşit dağılımda dataset'i birleştirilerek üstesinden gelinebilir.
neutrall_cnt, worry_cnt, happiness_cnt, sadness_cnt = 0, 0, 0, 0
love_cnt, surprise_cnt, fun_cnt, relief_cnt = 0, 0, 0, 0
hate_cnt, empty_cnt, enthusiasm_cnt, boredom_cnt, anger_cnt = 0, 0, 0, 0, 0

neutrall_correct, worry_correct, happiness_correct = 0, 0, 0
sadness_correct, love_correct, surprise_correct, fun_correct = 0, 0, 0, 0
relief_correct, hate_correct, empty_correct, enthusiasm_correct = 0, 0, 0, 0
boredom_correct, anger_correct = 0, 0

for x in range(len(X_val)):
    result = model.predict(X_val[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]
    if np.argmax(result) == np.argmax(Y_val[x]):
        if np.argmax(Y_val[x]) == 0:
            empty_correct += 1
            worry_correct += 2
            sadness_correct += 3
            hate_correct += 4
            boredom_correct += 5
            anger_correct += 6
        else:
            love_correct += 7
            enthusiasm_correct += 6
            happiness_correct += 5
            surprise_correct += 4
            fun_correct += 3
            relief_correct += 2
            neutrall_correct += 1
    if np.argmax(Y_val[x]) == 0:
        empty_cnt += 1
        worry_cnt += 2
        sadness_cnt += 3
        hate_cnt += 4
        boredom_cnt += 5
        anger_cnt += 6
    else:
        love_cnt += 7
        enthusiasm_cnt += 6
        happiness_cnt += 5
        surprise_cnt += 4
        fun_cnt += 3
        relief_cnt += 2
        neutrall_cnt += 1

positive_correct = love_correct + enthusiasm_correct + happiness_correct
positive_correct += surprise_correct + fun_correct + relief_correct + neutrall_correct
negative_correct = empty_correct + worry_correct + sadness_correct + hate_correct + boredom_correct + anger_correct
positive_cnt = love_cnt + enthusiasm_cnt + happiness_cnt + surprise_cnt + fun_cnt + relief_cnt + neutrall_cnt
negative_cnt = empty_cnt + worry_cnt + sadness_cnt + hate_cnt + boredom_cnt + anger_cnt

print("pos_acc", positive_correct / positive_cnt * 100, "%")
print("neg_acc", negative_correct / negative_cnt * 100, "%")

# -------------------------------Son----------------------------------------------------

# Deneysel Bölüm.
content = ['It is Not that The Wind is Blowing! It is What The Wind is Blowing! Happy New Year!!']
content = tokenizer.texts_to_sequences(content)
content = pad_sequences(content, maxlen=28, dtype='int32', value=0)
sentiment = model.predict(content, batch_size=1, verbose=2)[0]
print(" ")
print(" ")
if np.argmax(sentiment) == 0:
    print("Result: Neutral")
elif np.argmax(sentiment) == 1:
    print("Result: Worry")
elif np.argmax(sentiment) == 2:
    print("Result: Happiness")
elif np.argmax(sentiment) == 3:
    print("Result: Sadness")
elif np.argmax(sentiment) == 4:
    print("Result: Love")
elif np.argmax(sentiment) == 5:
    print("Result: Surprise")
elif np.argmax(sentiment) == 6:
    print("Result: Fun")
elif np.argmax(sentiment) == 7:
    print("Result: Relief")
elif np.argmax(sentiment) == 8:
    print("Result: Hate")
elif np.argmax(sentiment) == 9:
    print("Result: Empty")
elif np.argmax(sentiment) == 10:
    print("Result: Enthusiasm")
elif np.argmax(sentiment) == 11:
    print("Result: Boredom")
elif np.argmax(sentiment) == 12:
    print("Result: Anger")
