import torch
import streamlit as st
import numpy as np
from PIL import Image
from requests import get
from io import BytesIO
from torchvision import io

from model.model_1 import predict_1
from model.model_2 import predict_2

choice_model = None
image = None
push_model = False
video='images/IMG_9885.mp4'

st.write("""
 # Детекционные модели YOLOv5 & YOLOv8n🔎
""")

st.info("На странице предоставлены две модели детекции судов и кораблей на фотоснимках. \n"
        "Веса модели YOLOv5 сделаны Русланом, веса модели YOLOv8 - Романом. Какую палочку Twix выберешь ты, решать только тебе!🥰")

with st.sidebar:
    st.info('Обязательно выберите модель, которой хотите воспользоваться: \n')
    choice_model = st.radio('Выберите модель:', options=['YOLOv5', 'YOLOv8'], index=None)

    if choice_model is not None:
        st.info('Выберите каким способом хотите загрузить данные:')

        st.warning('📍ВНИМАНИЕ: Файл/картинку необходимо грузить только с расширением .jpg/jpeg')
        choice_image = st.radio('Выберите необходимый способ загрузки своей картинки:', options=['URL-адрес', 'Файл'], index=None)


        if choice_image == 'URL-адрес':
            image = st.text_input('Вставьте URL-адрес свой картинки для детекции')

            if image != '':

                response = get(image)
                image_url = Image.open(BytesIO(response.content))

        elif choice_image == 'Файл':
            image = st.file_uploader('Загрузите свою картинку для детекции через загрузчик файлов')

            if image is not None:

                image_url = Image.open(image)

        if image is not None and image != '':
            st.info('Не забывайте, что данные модели созданы детектировать только следующие объекты: '
                       '\n- Корабли \n- Судна \n- Лодки'
                       '\n\nОстальные элементы не будут распознаны на фотографии или же приведут к неверным результатам работы моделей!🥲')
            push_model = st.button('Нажмите, чтобы модель детектровала необходимые объект(ы) на картинке')


if image is not None and image != '':
    st.subheader('Вы загрузили следующую фотографию:')
    st.image(image, use_column_width=True, caption='Ваша загруженная фотография')

if push_model:
    if choice_model == 'YOLOv5':
        predict = predict_1(image_url)
        render_predict = predict.render()
        st.image(render_predict, use_column_width=True, caption='Фото с детектированным объектом(тами)')
    #  ТУТ ПРЕДСКАЗЫВАЕТ МОДЕЛЬ РУСЛАНА
    elif choice_model == 'YOLOv8':
        predict = predict_2(image_url)
        detect_img = Image.fromarray(predict)
        st.image(detect_img, use_column_width=True, caption='Фото с детектированным объектом(тами)')
    # ТУТ ПРЕДСКАЗЫВАЕТ МОДЕЛЬ РОМЫ

if st.checkbox('Нажмите, чтобы просмотреть видео детекцию модели YOLOv5📹'):

    st.info("Модель YOLO очень проста в использовании для задач детекции объектов на изображении. Однако, кажется, \
            что детекции объектов на видео сложна для данной модели. Но не тут то было!😏 \
            \nПри должном обучении и настройки весов под необходимую задачу модель YOLO способна очень быстро распознавать на видео нужные объекты.")

    st.write("""
     #### • На видео преставлена работа YOLOv5l
    """)

    st.video(video)

    st.warning("📍ВНИМАНИЕ: Если видео не показывается в Вашем браузере, попробуйте скачать его с GitHub'а проекта в папке /images \
    и просмотреть на своём компьютере!")
