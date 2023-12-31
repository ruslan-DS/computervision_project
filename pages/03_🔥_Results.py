import streamlit as st
from PIL import Image

import matplotlib.pyplot as plt

st.write("""
 ## Очищение документов от шума📝 Итоги.
""")

st.write("""
 ##### График Loss-функции
""")
st.image('images/Loss.png')

st.write("""
 ##### Сравнение результатов обработки изображений: Шумное изображение/Очищенное изображение/Цель
""")

st.image('images/noise1.png')
st.image('images/noise2.png')

"""
###### Итоги. Архитектура автоэнкодера состоит из энкодера и декодера, по 3 сверточных слоя в каждом, без применения пуллинга, \
В качестве функции потерь была использована L1Loss(), при обучении всего задействовано параметров: \
309 683. В качестве оптимайзера был использован Adam. Модель досточно сносно обучилась очищать черно-белые документы от неглубокого \
      шума. Не предназначена для работы с цветными изображениями, качество очистки хуже, на выходе получим ч/б изображение.
"""

st.write("""
 ## Детекционные модели YOLOv5 & YOLOv8n🔎
""")

st.write("""
 ### Результаты и выводы по работе с YOLOv8n
""")

st.text('Состав датасета')
st.text('Тренировочный сет - 19 396 размеченных изображений')
st.text('Валидационный сет - 2 165 размеченных изображений')
st.text('Тестовый сет - 1 573 размеченных изображения')
"""
##### Примеры изображений
"""
image_1 = Image.open('images/1.jpg')
image_2 = Image.open('images/2.jpg')
image_3 = Image.open('images/3.jpg')
image_4 = Image.open('images/4.jpg')
image_5 = Image.open('images/5.jpg')
image_6 = Image.open('images/6.jpg')
fig, axes = plt.subplots(2, 3, figsize=(3, 2))
axes[0][0].imshow(image_1)
axes[0][0].axis('off')
axes[0][1].imshow(image_2)
axes[0][1].axis('off')
axes[0][2].imshow(image_3)
axes[0][2].axis('off')
axes[1][0].imshow(image_4)
axes[1][0].axis('off')
axes[1][1].imshow(image_5)
axes[1][1].axis('off')
axes[1][2].imshow(image_6)
axes[1][2].axis('off')
st.pyplot(fig)

"""
##### Примеры уникальной разметки
"""

r_1 = Image.open('images/r_1.jpg')
r_2 = Image.open('images/r_2.jpg')
r_3 = Image.open('images/r_3.jpg')
r_4 = Image.open('images/r_4.jpg')
r_5 = Image.open('images/r_5.jpg')
r_6 = Image.open('images/r_6.jpg')
fig2, axes2 = plt.subplots(2, 3, figsize=(3, 2))
axes2[0][0].imshow(r_1)
axes2[0][0].axis('off')
axes2[0][1].imshow(r_2)
axes2[0][1].axis('off')
axes2[0][2].imshow(r_3)
axes2[0][2].axis('off')
axes2[1][0].imshow(r_4)
axes2[1][0].axis('off')
axes2[1][1].imshow(r_5)
axes2[1][1].axis('off')
axes2[1][2].imshow(r_6)
axes2[1][2].axis('off')
st.pyplot(fig2)

st.write("""
 ##### График Loss-функций
""")
st.image('images/yolo8_loss.png')
st.write("""
 ##### График изменения метрик
""")
st.image('images/yolo8_metrics.png')

st.write("""
 ##### Итоговые результаты
""")

st.image('images/yolo_8n_res.png')

"""
###### Итоги. В качестве модели была использована младшая модель yolo8 - 8nano\
Обучалась на урезанном датасете из 5000 изображений (3500-трейн,1500-валидация) в течение 186 эпох. \
Результат удовлетворительный, т.к. модель все ещё плохо детектирует при определенных ситуациях.\
Выправить ситуацию возможно при увеличении количества эпох обучений или переработке датасета и его разметки"""

st.write("""
 ### Результаты и выводы по работе с YOLOv5
""")

st.write("""
  #### Вывод по работе с рандомно сгенерированной выборкой:
  \n Результаты на двух моделях YOLOv5m и YOLOv5x, который обучались на выборках сгенерированных функцией, не лучшие.
  Изначальные проблемы с исходной выборкой и большая вероятность рандомно сгенерировать не репрезентативную выборку сильно повлияли на модель.
  Использование всей выборки на ещё меньшей модели (например, YOLOv5s) и оптимального количества обучающих эпох привели бы к лучшим результатам.
""")

st.image('images/3ru.jpg', use_column_width=True, caption='Результаты модели YOLOv5m на рандомной выборке размером 2500 изображений.')

st.write("""
 #### Вывод по работе с первой моделью YOLOv5l на всей выборке:
 \n Модель обучалась около 4 часов, так и не заверила до конца процесс обучения до 150 эпох. В целом, результат намного лучше, чем у моделей, 
 обучающихся не на всей выборке - видно положительное влияние полного количества данных.
 Модель показывает неплохие результаты, однако видно, что есть недообучение модели в связи редкими случаями некорректной работы.
 Данная модель остается ведущей по метрикам и показателям детектирования изображений из тестовой выборки.
""")

col1, col2 = st.columns(2)

with col1:
    st.image('images/2ru.jpg', caption='MAP Metric YOLOv5l на 100 эпохах.')

with col2:
    st.image('images/4ru.jpg', caption='Pr & Recall YOLOv5l на 100 эпохах')

st.image('images/1ru.jpg', caption='Batch из валидационной выборки YOLOv5l')


st.write("""
 #### Переобучение YOLOv5m:
 \n Первоначальное впечатление о том, что вся выборка окажет более положительное влияние на любую модель, сыграла со мной в злую шутку.
 Модель явно переобучилась, показатели после 100 эпохи стали только ухудшаться, а веса модели стали подстраиваться лишь под тренировочные данные.
 Остается сделатьв вывод, что на данном Датасете, который имеет очень странное распределение кораблей на изображениях, огромное количество эпох
 не будет оптимальным решением.
""")
col1, col2 = st.columns(2)

with col1:
    st.image('images/photo_2023-12-03_15-35-54.jpg', caption='Общие показатели обучения YOLOv5m на 200 эпохах.')

with col2:
    st.image('images/photo_2023-12-03_15-35-54 (2).jpg', caption='PR Curve модели YOLOv5m на 200 эпохах.')


st.write("""
   #### Конечный вывод по работе:
   \n За два рабочих дня, создав 6 дополнительных аккаунтов Google Colab и имея под рукой два рабочих Датасета, удалось запустить 4 раза обучение 
   на AutoEncoder с тестированием различной архитектуры модели. А также 6 раз обучения различных моделей YOLOv5, начиная с YOLOv5m, 
   заканчивая самой большой - YOLOv5x. Наилучший результат на данный момент показала модель YOLOv5L, которая обучалась оптимальных 100 эпох на всем Датасете.
   Именно данная модель остается лучшей среди всех испробованных, поэтому её веса будут содержатся в игрушечной модели YOLOv5L на странице "🚢 Ship".
   Всем спасибо! 😇
""")
