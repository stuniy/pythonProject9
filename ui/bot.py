import pickle
 # Подключаем модуль для Телеграма
import telebot
# Импортируем типы из модуля, чтобы создавать кнопки
from telebot import types
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# загрузка модели
model=pickle.load(open("model\predictor_model1.pkl",'rb'))


# загрузка данных
def iris_download():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Бинаризуйте вывод
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    return X, y


# подготовка данных
def iris_preprocess(X, y):
    # перемешать и разделить тренировочные и тестовые наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=2)
    return X_train, X_test, y_train, y_test


def iris_tree(X_train, X_test, y_train, y_test):
    # Обучаем предсказывать каждый класс по сравнению с другим
    clf = DecisionTreeClassifier()
    y_score = clf.fit(X_train, y_train)
    # Вычислить матрицу неточности
    predicted = y_score.predict(X_test)
    acc = accuracy_score(y_test, predicted)
    cnf = confusion_matrix(y_test.argmax(axis=1), predicted.argmax(axis=1))
    return acc, cnf


def iris_knear(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier()
    y_score = model.fit(X_train, y_train)
    predicted = y_score.predict(X_test)
    acc = accuracy_score(y_test, predicted)
    cnf = confusion_matrix(y_test.argmax(axis=1), predicted.argmax(axis=1))
    return acc, cnf

# Указываем токен
bot = telebot.TeleBot('1937629580:AAGisRGqMOzSkHcqBzU_XI2Eh0XmKz9t-uc')


# Метод, который получает сообщения и обрабатывает их
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    # Если написали «Привет»
    if message.text == "Привет":
        # Пишем приветствие
        bot.send_message(message.from_user.id, "Привет, сейчас я расскажу тебе что я умею делать.")
        bot.send_message(message.from_user.id, "Я умею классифицировать данные")
        # Готовим кнопки
        keyboard = types.InlineKeyboardMarkup()
        # По очереди готовим текст и обработчик
        key_oven = types.InlineKeyboardButton(text='Ирисы', callback_data='iris')
        # И добавляем кнопку на экран
        keyboard.add(key_oven)
        # Показываем все кнопки сразу и пишем сообщение о выборе
        mesg = bot.send_message(message.from_user.id, text='Выбери выборку', reply_markup=keyboard)
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши Привет")
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")
    # он ждёт сообщение пользователя и потом вызывает указанную функцию
    bot.register_next_step_handler(mesg, send_text)


def send_text(message):
    if message.text == "Обучение":
        # Готовим кнопки
        keyboard = types.InlineKeyboardMarkup()
        # По очереди готовим текст и обработчик для каждого знака зодиака
        key_oven = types.InlineKeyboardButton(text='Дерево решений', callback_data='tree')
        # И добавляем кнопку на экран
        keyboard.add(key_oven)
        # По очереди готовим текст и обработчик для каждого знака зодиака
        key_oven = types.InlineKeyboardButton(text='К-ближайших соседей', callback_data='kb')
        # И добавляем кнопку на экран
        keyboard.add(key_oven)
        # Показываем все кнопки сразу и пишем сообщение о выборе
        bot.send_message(message.from_user.id, text="Какой метод использовать?", reply_markup=keyboard)
    else:
        bot.send_message(message.from_user.id, "Напиши, Обучение")


# Обработчик нажатий на кнопки
@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    # Если нажали на одну из 12 кнопок — выводим гороскоп
    if call.data == "iris":
        # вызов функции
        global X, y, X_train, X_test, y_train, y_test
        X, y = iris_download()
        msg = str(X[0:5])
        msg = 'Наша выборка \n' + msg
        # Отправляем текст в Телеграм
        bot.send_message(call.message.chat.id, msg)
        X_train, X_test, y_train, y_test = iris_preprocess(X, y)
        mesg1 = bot.send_message(call.message.chat.id, "Напиши, Обучение")
    # bot.register_next_step_handler(mesg1, send_text)
    elif call.data == "tree":
        # вызов функции
        z, q = iris_tree(X_train, X_test, y_train, y_test)
        msg = f"Точность классификации: {round(z, 3) * 100}%"
        # Отправляем текст в Телеграм
        bot.send_message(call.message.chat.id, msg)
        msg3 = str(q[0:5])
        msg3 = 'Результат классификации \n' + msg3
        # Отправляем текст в Телеграм
        mesg3 = bot.send_message(call.message.chat.id, msg3)

    elif call.data == "kb":
        # X, y = iris_download()
        # X_train, X_test, y_train, y_test = iris_preprocess(X,y)
        # вызов функции
        z, q = iris_knear(X_train, X_test, y_train, y_test)
        msg = f"Точность классификации: {round(z, 3) * 100}%"
        # Отправляем текст в Телеграм
        bot.send_message(call.message.chat.id, msg)
        msg3 = str(q[0:5])
        msg3 = 'Результат классификации \n' + msg3
        # Отправляем текст в Телеграм
        mesg3 = bot.send_message(call.message.chat.id, msg3)
    else:
        bot.send_message(call.message.chat.id, "Выбери метод")


# Запускаем постоянный опрос бота в Телеграме
bot.polling(none_stop=True, interval=0)