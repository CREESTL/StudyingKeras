"""
Мы будем работать с набором данных Reuters — выборкой новостных лент и их тем, публиковавшихся агентством Reuters в 1986 году. Это простой набор данных, широко используемых для классификации текста. Существует 46 разных тем; некоторые темы более широко представлены, некоторые — менее, но для каждой из них в обучающем наборе имеется не менее 10 примеров.

ПОДКЛЮЧЕНИЕ ВСЕХ МОДУЛЕЙ
"""

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

"""ЗАГРУЗКА ДАННЫХ"""

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10_000)

print(f'number of training examples {len(train_data)}')
print(f'number of testing examples {len(test_data)}')

# каждый пример - список индексов слов
print(train_data[10])

"""ДЕКОДИРОВАНИЕ НОВОСТЕЙ ОБРАТНО В ТЕКСТ"""

def decode_words():
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_words = ' '.join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
    return decoded_words

decoded_words = decode_words()
print(decoded_words)

"""ВЕКТОРИЗАЦИЯ ДАННЫХ"""

def vectorize(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
    # в матрице значение из последовательности (sequence) становится единицей, а все остальное - нули
        results[i, sequence] = 1
    return results

x_train = vectorize(train_data)
x_test = vectorize(test_data)

"""ONE HOT КОДИРОВАНИЕ МЕТОК"""

def to_one_hot(labels, dimension=46):
      results = np.zeros((len(labels), dimension))
      for i, label in enumerate(labels):
        results[i, label] = 1
      return results

y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

'''
ТО ЖЕ САМОЕ МОЖНО СДЕЛАТЬ ТАК
'''
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

"""СОЗДАНИЕ МОДЕЛИ СЕТИ"""

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(64, activation='relu'))
# в выходном слое будет 46 нейронов
# это значит, что будет возвращаться 46-мерный вектор, где каждый элемент - отдельный выходной класс
# softmax будет возвращать вектор из 46 элементов (от 0 до 1), где каждый элемент будет показывать вероятность
# принадлежности входного образца к каждому из 46 классов
model.add(Dense(46, activation='softmax'))

"""КОМПИЛЯЦИЯ МОДЕЛИ"""

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

"""СОЗДАДИМ ВЫБОРКУ ИЗ 1000 ПРОВЕРОЧНЫХ ОБРАЗЦОВ"""

x_check = x_train[:1000]
partial_x_train = x_train[1000:]

y_check = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

"""ТРЕНИРОВКА И ПРОВЕРКА ТОЧНОСТИ ВО ВРЕМЯ ТРЕНИРОВКИ"""

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_check, y_check))
history = history.history

"""СОЗДАНИЕ ГРАФИКОВ ПО ДАННЫМ ТРЕНИРОВКИ"""

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



# рисуем все графики
draw_graph(history)

"""ПЕРЕОБУЧЕНИЕ ПРОСИХОДИТ НА 9-ОЙ ЭПОХЕ
УМЕНЬШИМ КОЛИЧЕСТВО ЭПОХ ДО 9
"""

history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_check, y_check))
history = history.history

draw_graph(history)

"""ОЦЕНКА ТОЧНОСТИ НА ТЕСТОВЫХ ДАННЫХ"""

model.evaluate(x_test, y_test)

"""ПРЕДСКАЗАНИЕ НА НОВЫХ ДАННЫХ"""

predictions = model.predict(x_test)

# убедимся, что выводятся показания для всех 46 классов
print(predictions[0].shape)

# наиболее вероятный элемент
np.argmax(predictions[0])