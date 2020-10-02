'''
Вы будете работать с набором данных IMDB: множеством из 50 000 самых
разных отзывов к кинолентам в интернет-базе фильмов (Internet Movie Database).
Набор разбит на 25 000 обучающих и 25 000 контрольных отзывов, каждый набор на
50 % состоит из отрицательных и на 50 % из положительных отзывов.
'''
from keras.datasets import imdb
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

# данные необходимо преобразовать в бинарную матрицу
# то есть например если есть массив [3,5], то он превратиться в вектор формы (n, 10 000), где
# все числа, кроме тех, которые на 3 и 5 позициях, будут нулями, а те - единицами.
def vectorize_sequences(sequences,dimension=10_000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# функция строит графики после тренировки
def draw_graph(history):
    loss_values = history["loss"]
    validation_loss_values = history["val_loss"]

    epochs = range(1, len(history['accuracy']) + 1)

    #               ГРАФИКИ ПОТЕРЬ
    # синими точками рисуется график потерь на этапе обучения
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    # синей линией рисуется график потерь на этапе проверки
    plt.plot(epochs, validation_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # очищаем рисунок
    plt.clf()

    #               ГРАФИКИ ТОЧНОСТИ
    acc_values = history['accuracy']
    validation_acc_values = history['val_accuracy']
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, validation_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# загружаем данные (предобработанные)
# num_words=10000 означает, что будет встречаться только 10к самых популярных слов, а
# редкие будут отброшены
# train_data и test_data — это списки отзывов;
# каждый отзыв — это список индексов слов (кодированное представление последовательности слов).
# train_labels и test_labels — это списки нулей и единиц, где нули соответствуют отрицательным
# отзывам, а единицы — положительным
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("[INFO] RAW DATA IS:")
print(f"train_data: \n{train_data}")
print(f"\ntest_data: \n{test_data}")
print(f"\ntrain_labels: \n{train_labels}")
print(f"\ntest_labels: \n{test_labels}")


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


# теперь создаем саму сеть
# пока что я не знаю сколько слоев использовать и сколько нейронов использовать в каждом слое
# это будет разбираться позднее
# сейчас создаю три слоя: 16 16 и 1 нейрон
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10_000, )))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # эта функция активации возвращает вероятность (от 0 до 1)

'''
Если бы в каждом слое были бы только линейне функции, то наложение слоев друг на друга
ничего бы не давало, так как функция оставалось бы линейной
А функция активации - не линейная. Поэтому она позволяет с увеличением количества слоев 
расширять пространство гипотез
'''
# модель настраивается с помощью оптимизатора rmsprop
# мы реалищуем бинарную классификацию, так что используем соответствующую функцию потерь loss
# компилируем модель
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# во время тренировки можно проверять точность модели на отдельной части тренировочных данных
# разделим тренировочные данные пополам
# данные для тренировки
partial_x_train = x_train[10_000:]
partial_y_train = y_train[10_000:]
# данные для проверки прямо во время тренировки
check_x_train = x_train[:10_000]
check_y_train = y_train[:10_000]

# возвращается history - объект, у которого есть поле - словарь .history, содержащий всю инфу
history = model.fit(partial_x_train, partial_y_train, validation_data=(check_x_train, check_y_train), epochs=20, batch_size=512)

history = history.history

# рисуем все графики
draw_graph(history)

# по графикам наблюдается переобучение
# на тренировке точность растет, а потеря падает
# на проверке после 4ой эпохи потеря начинает расти и точность падать
# сетка слишком "привыкла" к тренировочным данным и не может нормально распознать
# то, что никогда раньше не видела

# уменьшим число эпох до 4
history = model.fit(partial_x_train, partial_y_train, validation_data=(check_x_train, check_y_train), epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
# получем точность примерно 85%
print(f"[INFO] EVALUATION RESULTS ARE: {results}")

# попробуем предсказать вероятность, что отзывы будут положительные
# некоторые значения близяться к 1, другие к 0 - значит, что сетка понимает, положительный это отзыв
# или отрицательный
# некоторые, например 0.6 - значит, что сетка не может принять решение
predictions = model.predict(x_test)

print(f'\n[INFO] PREDICTIONS ARE: \n {predictions}')


