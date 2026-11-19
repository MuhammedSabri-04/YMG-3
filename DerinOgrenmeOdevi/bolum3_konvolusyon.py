# BÖLÜM 3 - Konvolüsyon işleminin kodlanması
# Amaç: Tek bir Conv2D katmanı oluşturup giriş/çıkış boyutlarını ve
# filtre ağırlıklarını incelemek.

import tensorflow as tf   # derin öğrenme kütüphanesi (Keras bunun içinde)
import numpy as np       # sayısal diziler (matris) için

# Rastgelelik kontrolü: rastgele üretilen her şey (ağırlıklar, giriş)
# her çalıştırmada aynı çıksın diye "tohum" değeri sabitlenir.
tf.keras.utils.set_random_seed(42)

# 3x3 boyutunda 16 filtreli konvolüsyon katmanı
conv = tf.keras.layers.Conv2D(
    filters=16,                     # 16 farklı filtre -> 16 çıkış kanalı
    kernel_size=(3, 3),             # her filtre 3x3 boyutunda
    activation='relu',              # sonuca ReLU uygulanır (negatifler 0 olur)
    kernel_initializer='he_normal'  # başlangıç ağırlıkları He yöntemiyle rastgele
)

# Örnek giriş görüntüsü: rastgele sayılardan oluşan sahte bir görüntü.
# Boyut sırası: (görüntü sayısı, yükseklik, genişlik, kanal) -> 1 adet 128x128 gri görüntü
dummy_input = tf.random.normal((1, 128, 128, 1))

# Konvolüsyon işlemi: filtreler görüntü üzerinde gezdirilir.
# Katman ilk kez burada çağrıldığı için ağırlıkları da burada oluşturulur.
output = conv(dummy_input)

# Ağırlıkları al: Conv2D katmanında iki parametre grubu vardır
# weights -> filtre değerleri, biases -> her filtrenin sabit (bias) değeri
weights, biases = conv.get_weights()

print("Giriş boyutu:", dummy_input.shape)              # (1, 128, 128, 1)
print("Çıkış boyutu:", output.shape)                   # (1, 126, 126, 16)
print("Filtre ağırlıklarının boyutu:", weights.shape)  # (3, 3, 1, 16)
print("Bias boyutu:", biases.shape)                    # (16,)

# İlk filtrenin ağırlıkları: [satır, sütun, giriş kanalı, filtre no]
# ":" o boyutun tamamını al demek -> 0. filtrenin 3x3 matrisi
print("İlk filtre:")
print(weights[:, :, 0, 0])


# ---------------------------------------------------------------
# Yapılacak çalışma 7: Bütün filtrelerin başlangıç ağırlıklarını
# görselleştiriniz.
# ---------------------------------------------------------------
import matplotlib.pyplot as plt   # grafik çizme kütüphanesi

# 4x4'lük bir ızgara oluştur: 16 filtrenin her biri için bir kutu
fig, axes = plt.subplots(4, 4, figsize=(8, 8), layout="constrained")

# axes.flat: 16 kutuyu sırayla verir; i = filtre numarası (0-15)
for i, ax in enumerate(axes.flat):
    filtre = weights[:, :, 0, i]  # i'nci filtrenin 3x3 ağırlıkları
    # Filtreyi resim gibi çiz: koyu = negatif, açık = pozitif ağırlık.
    # vmin/vmax tüm filtrelerde aynı renk ölçeği kullanılsın diye verildi.
    im = ax.imshow(filtre, cmap='gray',
                   vmin=weights.min(), vmax=weights.max())
    # Her hücreye ağırlık değerini yaz (koyu hücrede beyaz, açık hücrede siyah)
    for r in range(3):          # satır
        for c in range(3):      # sütun
            renk = 'white' if filtre[r, c] < 0 else 'black'
            ax.text(c, r, f"{filtre[r, c]:.2f}", ha='center', va='center',
                    color=renk, fontsize=8)
    ax.set_title(f"Filtre {i + 1}")
    ax.axis('off')              # eksen çizgilerini ve sayılarını gizle

# Sağ tarafa renk skalası ekle (hangi renk hangi değere karşılık geliyor)
fig.colorbar(im, ax=axes, shrink=0.6, label="Ağırlık değeri")
fig.suptitle("16 filtrenin başlangıç ağırlıkları (3x3, he_normal)")
plt.savefig("bolum3_filtreler.png")   # grafiği dosyaya kaydet
plt.show()                            # grafiği ekranda göster
