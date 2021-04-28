import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#data_name = input()
data_name = "supermarket_sales.csv"
article_read = pd.read_csv(data_name, delimiter=',')

# распределение набора данных по двум компонентам X и Y
dataset = article_read.values
X = dataset[:, 6:9]
y = dataset[:, 4]
# Разделение X и Y на
# Учебный набор и набор для тестирования
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

# выполнение препроцессорной части
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Применение функции PCA на тренировках
# и тестовый набор X-компонента
#pca = PCA(n_components=2)
#LLE = LocallyLinearEmbedding(n_components=2)
#Is = Isomap(n_components=2)
tSNE = TSNE(n_components=3)

#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)

#X_train = LLE.fit_transform(X_train)
#X_test = LLE.transform(X_test)

#X_train = Is.fit_transform(X_train)
#X_test = Is.transform(X_test)

X_train = tSNE.fit_transform(X_train)
X_test = tSNE.fit_transform(X_test)

# Подгонка логистической регрессии к тренировочному набору
classifier = LogisticRegression(random_state=0)
z = classifier.fit(X_train, y_train)

# Прогнозирование результата теста с использованием
# функция прогнозирования в LogisticRegression

y_pred = classifier.predict(X_test)

print('{:.2%}\n'.format(metrics.accuracy_score(y_test, y_pred)))
