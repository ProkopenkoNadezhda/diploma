import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from joblib import dump
from sklearn.metrics import plot_confusion_matrix

# функция открытия и считывания данных
def read():
    data_name = input("ВВЕДИТЕ НАЗВАНИЕ ФАЙЛА:  ")
    if data_name.endswith(".csv"):
        try:
            article_read = pd.read_csv(data_name, delimiter=',')
            return article_read, True
        except FileNotFoundError:
            print('!!   Файла с данным названием не существует  !!')
            return 0, False
    else:
        print("!!   Ошибка открытия файла, не верный тип файла  !!")
        return 0, False

# функция обработки входных данных
def prep(data, n):
    le = LabelEncoder()
    str = []
    for i in data.columns:
        if (data[i].dtype == "object") and i != data.columns[n]:
            str.append(i)
            data[i] = le.fit_transform(data[i].values)
            lists = le.classes_
            transform = le.transform(lists)
            d = {lists[i]: transform[i] for i in range(len(lists))}

    dataset = data.values
    X = data.drop(data.columns[n], axis=1).astype(float)
    y = dataset[:, n]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def model_LDA(X_train, X_test, y_train, y_test, n):
    LDA = LinearDiscriminantAnalysis(n_components=1)
    X_train = LDA.fit(X_train, y_train).transform(X_train)
    X_test = LDA.fit_transform(X_test, y_test)


    LDA_model = BaggingClassifier(LogisticRegression(tol=0.001), n_estimators=n)
    LDA_model.fit(X_train, y_train)

    # запись обученной модели в файл
    dump(LDA_model, "LDA_model.pkl")
    print('\n' + "Обучнная модель записана в LDA_model.pkl файл")

    y_pred = LDA_model.predict(X_test)
    with open("result_LDA.txt", "w") as file:
        file.write("Предсказанные значения:" + '\n' + str(y_pred) + '\n' +
                   "Тестовые значения:" + '\n' + str(y_test))
    print("Прогнозируемые значения и тестовые значения записаны в файл result_LDA.txt")

    score = LDA_model.score(X_test, y_test)
    print("Точность на тестовых данных LDA: {0:.2f} %".format(100 * score) + '\n')

    y = LabelEncoder().fit_transform(y_train)
    list = {y[i]: y_train[i] for i in range(len(y_train))}

    # график зависимости новоого признака
    plt.figure()
    colors = ['aqua', 'lime', 'red', 'g', 'b', 'yellow', 'c', 'purple']
    for color, i, target_name in zip(colors, list.keys(), list.values()):
        plt.scatter(X_train[y_train == list[i], 0], y_train[y_train == list[i]], color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.xlabel("Признак 1")
    plt.ylabel("Y")
    plt.title('LDA')
    plt.show()

    # визуализация матрицы путаницы
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(LDA_model, X_test, y_test)
    plt.show()

def model_PCA(X_train, X_test, y_train, y_test, n):
    from sklearn.decomposition import PCA
    PCA = PCA(n_components=2)

    PCA.fit(X_train)
    X_train = PCA.fit_transform(X_train)
    X_test = PCA.transform(X_test)

    PCA_model = BaggingClassifier(LogisticRegression(tol=0.001), n_estimators=n)
    PCA_model.fit(X_train, y_train)

    # запись обученной модели в файл
    dump(PCA_model, "PCA_model.pkl")
    print("Обучнная модель записана в PCA_model.pkl файл")

    y_pred = PCA_model.predict(X_test)
    with open("result_PCA.txt", "w") as file:
        file.write("Предсказанные значения:" + '\n' + str(y_pred) + '\n' +
                   "Тестовые значения:" + '\n' + str(y_test))
    print("Прогнозируемые значения и тестовые значения записаны в файл result_PCA.txt")

    score = PCA_model.score(X_test, y_test)
    print("Точность на тестовых данных PCA: {0:.2f} %".format(100 * score) + '\n')

    y = LabelEncoder().fit_transform(y_train)
    list = {y[i]:y_train[i] for i in range(len(y_train))}

    # график двухмерного представления данных
    plt.figure()
    colors = ['aqua', 'lime', 'red', 'g', 'b', 'yellow', 'c', 'purple']
    for color, i, target_name in zip(colors, list.keys(), list.values()):
        plt.scatter(X_train[y_train == list[i], 0], X_train[y_train == list[i], 1], color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.title('PCA')
    plt.show()

    # визуализация матрицы путаницы
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(PCA_model, X_test, y_test)
    plt.show()


while True:
    data, flag = read()
    if flag == False:
        break

    print("0 - Выход")
    print("1 - Выбор прогнзируемого параметра и обучение модели")
    tmp = input("Выберете действие  ")

    if tmp == '0':
        break
    elif tmp == '1':
        print("Список параметров датасета")
        for i in range(len(data.columns)):
            print('\t', data.columns[i], "-", i)
        index = input("Укажите номер прогнозируемого параметра:  ")
        X_train, X_test, y_train, y_test = prep(data, int(index))
        n = input("Введите количество базовых оценщиков в ансамбле:  ")
        model_LDA(X_train, X_test, y_train, y_test, int(n))
        model_PCA(X_train, X_test, y_train, y_test, int(n))
    else:
        print("!!   Введено неверное значение   !!", '\n')