# BÖLÜM 5 - ReLU ve Softmax fonksiyonlarının incelenmesi
# Amaç: İki aktivasyon fonksiyonunun sayılar üzerinde ne yaptığını görmek.

import tensorflow as tf

# ---------------------------------------------------------------
# A. ReLU:  ReLU(x) = max(0, x)
# Negatif sayılar 0 olur, pozitif sayılar aynen kalır.
# ---------------------------------------------------------------
x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])   # deneme sayıları

relu = tf.keras.layers.ReLU()   # ReLU katmanı oluştur
y = relu(x)                     # her sayıya ayrı ayrı uygula
print("Giriş:", x.numpy())      # .numpy() -> tensörü normal diziye çevirir
print("ReLU çıkışı:", y.numpy())

# ---------------------------------------------------------------
# B. Softmax:  softmax(z_i) = e^(z_i) / (tüm e^(z_j) toplamı)
# Ham skorları (logit) toplamı 1 olan olasılıklara çevirir.
# ---------------------------------------------------------------
logits = tf.constant([[2.0, 1.0, 0.1]])   # 3 sınıfın ham skorları
softmax = tf.keras.layers.Softmax()       # Softmax katmanı oluştur
probabilities = softmax(logits)           # skorları olasılığa çevir
print("Softmax sonuçları:")
print(probabilities.numpy())
print("Olasılıkların toplamı:")
print(tf.reduce_sum(probabilities).numpy())   # reduce_sum -> tüm elemanları topla