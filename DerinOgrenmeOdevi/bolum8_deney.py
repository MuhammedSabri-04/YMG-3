# BÖLÜM 8 - Küçük bir deney
# Deney 1: İlk konvolüsyon katmanında 8 filtre
# Deney 2: İlk konvolüsyon katmanında 16 filtre
# Diğer bütün parametreler Bölüm 6 ile aynıdır.

import time   # eğitim süresini ölçmek için
import tensorflow as tf
import numpy as np

# Veri seti (Bölüm 6 ile aynı): yükle, 0-1 arasına getir, kanal boyutu ekle
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]


# Aynı modeli iki kez kuracağımız için bir fonksiyon yazdık.
# Fonksiyona verilen sayı, ilk Conv2D katmanının filtre sayısı olur.
def model_olustur(ilk_filtre_sayisi):
    # Bölüm 6'daki model; sadece ilk Conv2D'nin filtre sayısı değişiyor
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(ilk_filtre_sayisi, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


# Her deneyin sonuçları burada saklanır: {8: {...}, 16: {...}}
sonuclar = {}

# Döngü iki kez döner: önce 8 filtre (Deney 1), sonra 16 filtre (Deney 2)
for filtre_sayisi in [8, 16]:
    print(f"\n===== İlk konvolüsyon katmanında {filtre_sayisi} filtre =====")

    # İki deney de aynı rastgele başlangıçla başlasın (adil karşılaştırma)
    tf.keras.utils.set_random_seed(42)

    # Modeli kur ve Bölüm 6'daki ayarlarla derle
    model = model_olustur(filtre_sayisi)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Süre ölçümü: eğitimden önce ve sonra saati al, farkı = eğitim süresi (sn)
    baslangic = time.time()
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )
    egitim_suresi = time.time() - baslangic

    # Test verisiyle değerlendir
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # Sonuçları kaydet; [-1] listenin son elemanı = son epoch'un doğruluğu
    sonuclar[filtre_sayisi] = {
        "egitim_dogrulugu": history.history['accuracy'][-1],  # son epoch
        "test_dogrulugu": test_accuracy,
        "sure": egitim_suresi,
    }

# Sonuç tablosu: ödevdeki tablo biçiminde (Markdown) yazdırılır,
# doğrudan rapora kopyalanabilir. :.4f -> virgülden sonra 4 basamak
d1, d2 = sonuclar[8], sonuclar[16]
print("\n| Ölçüt | Deney 1 | Deney 2 |")
print("|---|---|---|")
print("| Filtre sayısı | 8 | 16 |")
print(f"| Eğitim doğruluğu | {d1['egitim_dogrulugu']:.4f} | "
      f"{d2['egitim_dogrulugu']:.4f} |")
print(f"| Test doğruluğu | {d1['test_dogrulugu']:.4f} | "
      f"{d2['test_dogrulugu']:.4f} |")
print(f"| Eğitim süresi | {d1['sure']:.1f} sn | {d2['sure']:.1f} sn |")
