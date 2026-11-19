# BÖLÜM 4 - Pooling işleminin incelenmesi
# Amaç: 4x4'lük küçük bir matrise 2x2 Max Pooling uygulayıp
# her 2x2 bölgeden en büyük sayının seçildiğini görmek.

import tensorflow as tf

# Örnek giriş matrisi (4 satır, 4 sütun)
input_data = tf.constant([
    [[1., 3., 2., 4.],
     [5., 6., 1., 2.],
     [7., 2., 8., 3.],
     [4., 1., 5., 9.]]
])

# Boyutları düzenle: Batch, Height, Width, Channels
# Keras katmanları girişi bu 4 boyutlu biçimde bekler:
# 1 görüntü, 4 yükseklik, 4 genişlik, 1 kanal
input_data = tf.reshape(input_data, (1, 4, 4, 1))

pool = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),   # her seferinde 2x2'lik bir bölgeye bak
    strides=2           # pencereyi 2 adım kaydır -> bölgeler üst üste binmez
)

# Pooling uygula: her 2x2 bölgeden en büyük değer alınır -> sonuç 2x2
output = pool(input_data)

# [0, :, :, 0] -> 1. görüntünün 1. kanalını 2 boyutlu matris olarak yazdır
print("Giriş:")
print(input_data.numpy()[0, :, :, 0])

print("Pooling sonucu:")
print(output.numpy()[0, :, :, 0])
