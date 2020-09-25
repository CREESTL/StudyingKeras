'''
Здесь я изучаю операции с тензорами
'''

import numpy as np
'''
Тензоры можно складывать, перемножать и т.д.

Например в слое Dense с функцией relu

output = relu(dot(W, input) + b), где W - 2мерный тензор, a, b - векторы
'''

'''
BLAS (Basic Linear Algebra Subprograms) - это набор низкоуровневых процедур для вычислений с тензорами
'''

# поэлементные операции выполняются почти мгновенно
x = np.array([1,2,3,4,5])
y = np.array([6,7,8,9,10])
z = x + y
print(z)
# функция relu (заисывается как max(z,0) в аглебре)
z = np.maximum(z, 0.)
print(z)


'''
Когда складываются два тензора с РАЗНЫМИ формами, то меньший тензор расширяется до формы большего
Например, если сккладывать x = (10, 32) и y = (32,), то y расширится до (10,32) с добавлением новой оси
'''
# так создаются массивы случайных чисел в numpy
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32,10))
z = np.maximum(x ,y) # y расширяется до размеров x

'''
Скалярное произведение векторов
В результате получаеся ЧИСЛО (скаляр)
Могут перемножаться только векторы с ОДИНАКОВЫМ количеством элементов
Могут перемножаться только матрицы, с СИММЕТРИЧНОЙ формой (3,2) на (2,3) 
'''
# x.shape[1] = y.shape[0]
x = np.random.random((3,2))
y = np.random.random((2,3))
# у z будет форма (x.shape[0], y.shape[1])
z = np.dot(x, y)
print(f"scalar product of {x} and {y} is {z}")

'''
Изменение формы тензора
'''

x = np.array([[0, 1],
              [2, 3],
              [4, 5]])
x = x.reshape((6, 1))
print(f"reshaped x is")
print(x)
x = x.transpose()
print("transposed x is ")
print(x)

'''
Представим вектор A = [0.5, 1]. Его можно представить на коорд.плоскости
Чтобы его повернуть на угол h, то нужно его умножить на матрицу R = [u, v]
u = [cos(h), sin(h)], v = [-sin(h), cos(h)]
'''