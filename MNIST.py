from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# train - тренировочные данные
# после обучения модель будет проверяться на test

# изображения - в массивах numpy, метки - массив от 0 до 9
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"train_images.shape is {train_images.shape}")
print(f"length of labels is {len(test_labels)}")

# создаем сеть
'''
Каждый слой принимает данные и выводит их в более полезной форме. Набор слоев действует, как сито
Произвоится очистка данных
'''
network = models.Sequential()
# добавляется полносвязный слой
# на первом слое важно указвать размерность входных данных
network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
# добавляется слой потерь, он возвращает массив с 10 оценками вероятностей принадлежности рукописной
# цифры к классу
# все слои после первого сами подстраиваются по размерность данных
network.add(layers.Dense(10, activation="softmax"))

'''
Функция потерь - определяет как сеть должна оценивать качество свой работы и как корректировать ее в 
правильном направлении

Оптимизатор - механизм, с помощью которого сеть обновляет себя, опираясь на функцию потерь

Метрики для мониторинга - здесь интересует только точность (доля правильно классифицированных изображений)
'''

# компиляция сети
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

'''
Перед обучением необхдимо предобработать данные. Мы приводим их к такому виду, который нейронка ожидает принять
Изображения храняться в типе uint8 (от 0 до 255), преобразуем их в float32 (от 0 до 1)
'''

# предобработка данных
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255 # делим на 255, чтобы было от 0 до 1 значение

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# кодируем метки категорий (подробнее - глава 3)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# начинаем обучение сети
# fit() старается адаптировать модель под обучающие данные
'''
В процессе обучения отображаются:
1) loss - потеря на обучающих данных
2) accuracy - точность на обучающих данных
'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# проверим, как модель распознает контрольный набор
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(f"test accuracy is {test_acc}")
'''
Здесь точность чуть меньше, чем на тренировочном наборе. Это произошло из-за переобучения. Модель показывает
худшую точность на новом наборе, чем на тренировочном.
'''