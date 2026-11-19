# BÖLÜM 6 - Basit bir CNN modeli oluşturulması (Fashion-MNIST)
# Amaç: Giysi görüntülerini 10 sınıfa ayıran küçük bir CNN kurup eğitmek.

# ---------------------------------------------------------------
# A. Veri setinin yüklenmesi
# ---------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Her çalıştırmada aynı sonuç çıksın diye rastgelelik sabitlendi
# (Bölüm 3'teki gibi). Bölüm 8'deki Deney 1 de aynı sonucu verir.
tf.keras.utils.set_random_seed(42)

# Veri setini indir/yükle:
# x_ -> görüntüler (28x28 piksel), y_ -> etiketler (0-9 arası sınıf numarası)
# train -> 60.000 eğitim görüntüsü, test -> 10.000 test görüntüsü
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)
# Normalizasyon: piksel değerleri 0-255 arası, 255'e bölünce 0-1 arası olur.
# Küçük sayılarla eğitim daha hızlı ve kararlı olur.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# Kanal boyutu ekleme: (60000, 28, 28) -> (60000, 28, 28, 1)
# Conv2D katmanı kanal boyutunu bekler (gri görüntü = 1 kanal)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)

# ---------------------------------------------------------------
# B. CNN mimarisinin oluşturulması
# Sequential: katmanlar sırayla dizilir, birinin çıkışı diğerinin girişidir.
# ---------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),   # giriş: 28x28 gri görüntü
    # Birinci konvolüsyon: 8 filtre, 3x3 -> çıkış 26x26x8
    tf.keras.layers.Conv2D(
        8, (3, 3),
        activation='relu'
    ),
    # Birinci pooling: boyutu yarıya indirir -> 13x13x8
    tf.keras.layers.MaxPooling2D((2, 2)),
    # İkinci konvolüsyon: 16 filtre, 3x3 -> çıkış 11x11x16
    tf.keras.layers.Conv2D(
        16, (3, 3),
        activation='relu'
    ),
    # İkinci pooling: -> 5x5x16
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Çok boyutlu özellikleri tek boyuta dönüştür: 5*5*16 = 400 sayı
    tf.keras.layers.Flatten(),
    # Tam bağlantılı katman: 64 nöron, özellikleri birleştirip karar verir
    tf.keras.layers.Dense(64, activation='relu'),
    # 10 sınıf için çıktı: softmax her sınıf için bir olasılık verir
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()   # katmanları, çıkış boyutlarını ve parametre sayılarını yazdır

# ---------------------------------------------------------------
# C. Modelin derlenmesi ve eğitilmesi
# ---------------------------------------------------------------
model.compile(
    optimizer='adam',                        # ağırlıkları güncelleyen yöntem
    loss='sparse_categorical_crossentropy',  # hata fonksiyonu (etiketler 0-9 sayı olduğu için "sparse")
    metrics=['accuracy']                     # başarı ölçütü: doğru tahmin oranı
)

# Eğitim: history, her epoch'taki accuracy/loss değerlerini saklar
history = model.fit(
    x_train,
    y_train,
    epochs=5,              # tüm eğitim verisinin üzerinden 5 kez geç
    batch_size=64,         # ağırlıkları her 64 görüntüde bir güncelle
    validation_split=0.1   # eğitim verisinin %10'unu doğrulama için ayır
)

# Modeli daha önce hiç görmediği test verisiyle değerlendir
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ---------------------------------------------------------------
# D. Eğitim sonuçlarının görselleştirilmesi
# Solda accuracy, sağda loss grafiği; her birinde eğitim ve doğrulama eğrisi
# ---------------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))   # 10x4 inç boyutunda bir pencere

plt.subplot(1, 2, 1)   # 1 satır, 2 sütun -> 1. grafik (sol)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()           # hangi çizginin ne olduğunu gösteren kutu

plt.subplot(1, 2, 2)   # 2. grafik (sağ)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()   # grafiklerin birbirine taşmaması için boşlukları ayarla
plt.show()
