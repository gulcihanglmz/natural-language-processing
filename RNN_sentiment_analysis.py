import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from scipy.signal import max_len_seq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

from transformers_based_text__representation import tokenizer

data = {
    "text": [
        "Yemek çok tuzluydu, yiyemedim.",
        "Tatlılar taptazeydi, özellikle cheesecake enfesti.",
        "Servis çok yavaş, 40 dakika bekledik.",
        "Garson çok ilgiliydi, memnun kaldım.",
        "Mekanın dekorasyonu çok şıktı.",
        "İçecekler sıcak geldi, buz istememize rağmen yoktu.",
        "Sunum gerçekten şıktı, göze hitap ediyor.",
        "Masamız temizlenmemişti, önceki müşterinin artıkları vardı.",
        "Fiyat/performans açısından çok başarılıydı.",
        "Tatlılar aşırı şekerliydi.",
        "Servis hızlıydı, yemekler sıcaktı.",
        "Kahve acıydı, içemedim.",
        "Pizzaları taş fırında yapıyorlar, harikaydı.",
        "Rezervasyonsuz gitmeme rağmen hemen yer buldum.",
        "Siparişimiz eksik geldi, birkaç şey unutulmuştu.",
        "Et çok güzel marine edilmişti.",
        "Menüdeki birçok şey kalmamıştı.",
        "Doğum günü kutlamamızda çok ilgilendiler.",
        "Lavabolar çok kirliydi.",
        "Tatlar çok dengeliydi, ne eksik ne fazla.",
        "Tatlıların şekeri tam kararındaydı.",
        "Masada peçete yoktu, istemek zorunda kaldık.",
        "Porsiyonlar çok küçüktü, doymadım.",
        "Sunumlar Instagramlık, çok estetikti.",
        "Girişte kimse ilgilenmedi.",
        "Mekanda sigara içiliyordu, rahatsız olduk.",
        "Fiyatlar oldukça makul ve lezzetliydi.",
        "Tatlılar bayattı, hiç beğenmedim.",
        "Garson menüyü detaylıca anlattı.",
        "Servis sırasında içecek döküldü, özür bile dilenmedi.",
        "Bahçesi çok keyifliydi, ferah bir ortam.",
        "Et çiğ kalmıştı, geri göndermek zorunda kaldım.",
        "Yemekler harikaydı, tekrar gelmek isterim.",
        "Çatal-bıçak lekeli geldi.",
        "Rezervasyon süreci çok kolaydı.",
        "Masa çok sallanıyordu, rahatsız ediciydi.",
        "Salatanın içinde sinek vardı.",
        "Kapanışta minik hediyeler vermeleri çok nazikti.",
        "Tatlar uyumsuzdu.",
        "Menüdeki çeşitlilik harikaydı.",
        "İçeride sigara içiliyordu, duman altı kaldık.",
        "Açık mutfak sistemi güven veriyor.",
        "Garsonlar ilgisizdi, uzun süre bekledik.",
        "Porsiyonlar gayet doyurucuydu.",
        "İkram olarak çay ve lokum verdiler.",
        "Sandalyeler rahatsızdı.",
        "Mekanda sigara içilmeyen alanların olması güzel.",
        "Patatesler bayattı, çıtırlıktan eser yoktu.",
        "Tatlılar efsane, özellikle profiterol.",
        "Kendi baharatlarını kullanmaları fark yaratıyor.",
        "Menü çok yaratıcı ve farklıydı.",
        "İkram sözü verildi ama getirmediler.",
        "Garsonlar güler yüzlüydü.",
        "Yemekler geç geldi ve soğuktu.",
        "Hizmet kalitesi çok düşüktü.",
        "Tatlıların tadı damağımda kaldı.",
        "İç mekan dekorasyonu çok modern ve rahatlatıcıydı.",
        "Sipariş karıştı, başka masanın yemeği geldi.",
        "Fiyatlar çok yüksekti, beklentimin altında kaldı.",
        "Garsonlar çok bilgiliydi.",
        "Mekanın havası çok basıktı, klima çalışmıyordu.",
        "Yemeklerin tadı çok sıradandı.",
        "Atmosfer çok hoştu, müzikler rahattı.",
        "Servis elemanları kaba davrandı.",
        "Menüde yazan içerikle gelen yemek uyuşmuyordu.",
        "Kahvaltı serpmesi efsaneydi, çok çeşit vardı.",
        "Çocuklar için oyun alanı olması çok iyi.",
        "Mekanda garip bir koku vardı.",
        "Hesapta yanlışlık vardı, fazla ücret alındı.",
        "Kahve çok lezzetliydi, yanında gelen kurabiye de güzeldi.",
        "Alerjimi belirttiğim halde yanlış yemek geldi.",
        "Her şey çok taze ve sıcaktı.",
        "Yemek sonrası kahve ikramı hoş bir sürprizdi.",
        "Çalışanlar birbirleriyle tartışıyordu.",
        "Masamız çok kirliydi, hijyen kötüydü.",
        "Müzikler çok hoştu, ambiyans mükemmeldi.",
        "Yemeklerin rengi bile iştah kaçırıyordu.",
        "Lezzetler damağımda kaldı.",
        "Servis elemanları çok yardımseverdi.",
        "Beklentimi hiç karşılamadı.",
        "Her şey taze ve organikti.",
        "Garson yüzümüze bile bakmadı.",
        "Menü çok zayıftı.",
        "Tatlılar bayat ve tatsızdı.",
        "Et tam pişmemişti, yiyemedim.",
        "Garsonlar çok güler yüzlüydü.",
        "Mekan çok temizdi ve hijyen kurallarına uyuluyordu.",
        "Yemekler tam kararında pişmişti.",
        "İçecek menüsü çok zayıftı.",
        "Aileyle gidilecek nezih bir yer.",
        "Yemeklerin kokusu çok kötüydü.",
        "Garson menüyü bilmiyordu.",
        "Rezervasyonumuz hemen bulundu, beklemeden oturduk.",
        "Her şey taze ve organikti.",
        "Mekanın dışı çok bakımsızdı.",
        "İlgisiz personel yüzünden bir daha gitmem.",
        "Yemekten sonra midem bulandı.",
        "Servis berbattı, kimse ilgilenmedi.",
        "Garson menüyü detaylıca anlattı.",
        "İkram ettikleri çorba çok lezzetliydi.",
        "Doğum günümüz için süsleme bile yaptılar.",
        "Manzara eşliğinde akşam yemeği harikaydı."
    ],
    "label": [
        "negative", "positive", "negative", "positive", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "positive", "negative", "negative", "positive", "negative",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "positive", "negative", "positive",
        "negative", "positive", "positive", "negative", "positive",
        "positive", "negative", "negative", "positive", "positive",
        "negative", "negative", "positive", "negative", "negative",
        "positive", "negative", "negative", "positive", "negative",
        "negative", "positive", "negative", "negative", "positive",
        "negative", "negative", "positive", "negative", "negative",
        "positive", "positive", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "negative", "positive", "positive", "positive", "negative",
        "negative", "negative", "negative", "negative", "negative",
        "positive", "positive", "positive", "positive", "positive",
        "positive", "positive"
    ]
}

print(len(data["text"]))
print(len(data["label"]))

df = pd.DataFrame(data)
print(df)


# Tokenizer sınıfını başlatıyoruz
tokenizer = Tokenizer()

# 'text' sütunundaki metinler üzerinde tokenizer'ı eğitiyoruz (fit ediyoruz)
tokenizer.fit_on_texts(df["text"])

# DataFrame'deki 'text' sütunundaki her metni sayısal dizilere dönüştürüyoruz
# text_to_sequences(): Verilen metni (her bir kelimeyi) indekslerine dönüştürür
sequences = tokenizer.texts_to_sequences(df["text"])  # Correcting method name here

# Tokenizer tarafından oluşturulan kelime indeksini alıyoruz (kelimeleri sayısal indexlere eşliyoruz)
word_index = tokenizer.word_index

print(sequences)
print(word_index)

# padding
# En uzun dizinin uzunluğunu buluyoruz
maxlen = max(len(seq) for seq in sequences)  # Tüm cümleler içinde en uzun olanın kelime sayısını al

# Tüm dizileri aynı uzunlukta olacak şekilde pad (doldurma) yapıyoruz
x = pad_sequences(sequences, maxlen)  # Daha kısa dizilerin başına 0 eklenerek hepsi aynı uzunluğa getiriliyor

# Sonucun boyutunu yazdırıyoruz: (örnek sayısı, sabit dizi uzunluğu)
print(x.shape)  # Örn: (102, 8) -> 102 tane cümle, her biri 8 uzunluğunda sabitlenmiş


# label encoding
label_encoder = LabelEncoder()
y = label_encoder. fit_transform(df["label"])

# train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# metin temsili: word embedding: word2vec
sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=50, window = 5, min_count=1)

embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

'''
Word2Vec modelini eğitiyoruz:
vector_size=50: Her kelime 50 boyutlu bir vektörle temsil edilecek.
window=5: Bir kelimenin çevresine 5 kelimeye kadar bakarak bağlam öğrenir.
min_count=1: 1'den az geçen kelimeler yok sayılmaz (her kelime öğrenilsin).

Her kelimenin Word2Vec vektörünü embedding matrisine yazıyoruz.
word_index.items() → {“tatlı”: 5, “çok”: 3, ...}
Eğer Word2Vec modeli bu kelimeyi öğrenmişse (if word in word2vec_model.wv)
O kelimenin vektörünü embedding_matrix[i]’ye koyuyoruz.

'''

model = Sequential() # Modelimizi sıfırdan başlatıyoruz.
# embedding
model.add(Embedding(
    input_dim=len(word_index) + 1,      # Kelime sayısı (+1 çünkü indexler 1'den başlıyor)
    output_dim=embedding_dim,           # Her kelimenin vektör boyutu (50)
    weights=[embedding_matrix],         # Word2Vec'ten gelen embedding matrisini veriyoruz
    input_length=maxlen,                # Her cümlede kaç kelime olacağını belirtiyoruz (pad edilmiş haliyle)
    trainable=False                     # Embedding'leri değiştirme, olduğu gibi kullan (dondur)
))

# RNN layer
model.add(SimpleRNN(50,return_sequences=False))

# output layer
model.add(Dense(1,activation="sigmoid"))

# compile model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])

# train model
model.fit(X_train, y_train, epochs=10, batch_size = 2, validation_data=(X_test, y_test))

'''
# return_sequences=False: Sadece son gizli durumu alıyoruz 
(çünkü classification yapacağız, zaman serisinin tamamına ihtiyaç yok).

Binary classification olduğu için Dense(1) ve sigmoid kullanıyoruz (çıkış 0 ile 1 arasında)

adam: etkili bir optimizer.
binary_crossentropy: Binary classification'da kullanılır.
accuracy: Doğruluğu izliyoruz.

'''

# evaluate rnn model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# %% cumle siniflandirma calismasi
def classify_sentence(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen = maxlen)

    prediction = model.predict(padded_seq)

    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"

    return label

sentence = "Restaurant çok temizdi ve yemekler çok güzeldi"

result = classify_sentence(sentence)
print(f"Result: {result}")
