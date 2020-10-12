# -*- coding: utf-8 -*-
"""IMDB_films_v2
ЛУЧШИЙ способ избежать переобучения - увеличить набор тренировочных данных

САМЫЙ ПРОСТОЙ способ - уменьшить размерность сети. Это уменьшит количество изучаемых признаков. Уменьшиться и размер памяти, в который модель может сохранять "шаблоны", что орицательно влияет на общность.

Но если уменьшить слишком сильно, то модель будет недообучена. Важно найти компромисс между слишком большой и недостающе емкостью модели.

УСТАНОВКА ВСЕХ МОДУЛЕЙ
"""

from keras.datasets import imdb
from keras.layers import Dense, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras import regularizers
import numpy as np

"""ЧТЕНИЕ ДАННЫХ"""

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

"""ВЕКТОРИЗАЦИЯ ДАННЫХ"""

# данные необходимо преобразовать в бинарную матрицу
# то есть например если есть массив [3,5], то он превратиться в вектор формы (n, 10 000), где
# все числа, кроме тех, которые на 3 и 5 позициях, будут нулями, а те - единицами.
def vectorize_sequences(sequences,dimension=10_000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# векторизуем метки
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print("\n\n[INFO] PREPROCESSED DATA IS:")
print(f"train_data: \n{x_test}")
print(f"\ntest_data: \n{x_test}")
print(f"\ntrain_labels: \n{y_train}")
print(f"\ntest_labels: \n{y_test}")

"""СОЗДАНИЕ МОДЕЛИ №1"""

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))# выводит вероятность

"""СОЗДАНИЕ МОДЕЛИ №2 (С МЕНЬШЕЙ ЕМКОСТЬЮ)"""

model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(10000,)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

"""СОЗДАНИЕ МОДЕЛИ №3 (С ОЧЕНЬ БОЛЬШОЙ ЕМКОСТЬЮ)"""

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(10000,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

"""КОМПИЛЯЦИЯ МОДЕЛИ"""

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

"""ТРЕНИРОВКА МОДЕЛИ"""

partial_x_train = x_train[10_000:]
partial_y_train = y_train[10_000:]

check_x_train = x_train[:10_000]
check_y_train = y_train[:10_000]

history = model.fit(partial_x_train, partial_y_train, validation_data=(check_x_train, check_y_train), epochs=20, batch_size=512)

old_history = history.history

"""ФУНКЦИЯ РИСУЕТ ГРАФИКИ"""

# функция строит графики после тренировки
def draw_graph(history):
    loss_values = history["loss"]
    validation_loss_values = history["val_loss"]

    epochs = range(1, len(history['loss']) + 1)

    #               ГРАФИКИ ПОТЕРЬ
    # синими точками рисуется график потерь на этапе обучения
    plt.plot(epochs, loss_values, 'or', label='Training loss')
    # синей линией рисуется график потерь на этапе проверки
    plt.plot(epochs, validation_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # очищаем рисунок
    plt.clf()

    #               ГРАФИКИ ОШИБКИ
    acc_values = history['accuracy']
    validation_acc_values = history['val_accuracy']
    plt.plot(epochs, acc_values, 'or', label='Training accuracy')
    plt.plot(epochs, validation_acc_values, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()



# рисуем все графики
draw_graph(old_history)

"""ЧЕМ МЕНЬШЕ МОДЕЛЬ, ТЕМ ДОЛЬШЕ НЕ НАСТУПАЕТ ПЕРЕОБУЧЕНИЕ

ДОБАВИМ РЕГУЛЯРИЗАЦИЮ ВЕСОВ
"""

# Простая модель - это модель, в которой помимо всего прочего распределение весов 
# будет равномерным. Чтобы оно стало равномерным, необходимо ограничить весовые
# коэффициенты. Для этого используются регуляризаторы 2ух видов:
# L1 регуляризация - вводится штраф за увеличение весов, пропорциональный модулям
# весов
# L2 регуляризация - вводится штраф за увеличение весов, пропорциональный квадратам
# весов

model = Sequential()
# l2(0.001) значит, что каждый коэффициент в матрице весов слоя будет добавлять 
# 0.001 от самого себя в общее значение потерь сети
model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

"""КОМПИЛЯЦИЯ МОДЕЛИ"""

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

"""ТРЕНИРОВКА МОДЕЛИ"""

history = model.fit(partial_x_train, partial_y_train, validation_data=(check_x_train, check_y_train), epochs=20, batch_size=512)
history = history.history

"""ФУНКЦИЯ ВЫВОДИТ ГРАФИКИ ДВУХ МОДЕЛЕЙ"""

# функция строит графики после тренировки
def draw_graph(history_1, history_2):
    validation_loss_values_1 = history_1["val_loss"]

    epochs = range(1, len(history_1['loss']) + 1)

    #               МОДЕЛЬ 1
    #               ГРАФИКИ ПОТЕРЬ - СИНИМ
    plt.plot(epochs, validation_loss_values_1, 'b', label='m1_val_loss')
    plt.title('Training and validation loss OLD MODEL')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')


    #               МОДЕЛЬ 2 
    #               ГРАФИКИ ПОТЕРЬ - КРАСНЫМ
    validation_loss_values_2 = history_2["val_loss"]
    plt.plot(epochs, validation_loss_values_2, 'r', label='m2_val_loss')
    plt.title('Training and validation loss NEW MODEL')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.legend()
    plt.show()


# рисуем все графики
draw_graph(old_history, history)

"""ЗАМЕТНО, ЧТО ПОСЛЕ ДОБАВЛЕНИЯ РЕГУЛЯРИЗАЦИИ ПОТЕРЯ НА ВАЛИДАЦИИ РАСТЕТ МЕДЛЕНЕЕ, ЧТО ГОВОРИТ О БОЛЕЕ МЕДЛЕННОМ ПЕРЕОБУЧЕНИИ

ДОБАВЛЕНИЕ ПРОРЕЖИВАНИЯ
"""

# прореживание - обнуление случайно выбранных признаков на обучениии
# это помогает справиться с переобучением
# введение "шума" в выходные значения может разбить "шаблоны", которые создаются
# в сетке припереобучении
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dropout(0.5)) # коэффициент показывает сколько процентов обнуляется
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train, validation_data=(check_x_train, check_y_train), epochs=20, batch_size=512)
history = history.history

draw_graph(old_history, history)
