# Project - Computer Vision by YOLO and Autoencoder.

### Чтобы воспользоваться созданным приложением, перейдите по [ссылке](https://computervision-yolo.streamlit.app).

#### Проект был создан внутри крутой команды начинающих Data Scientis'ов - [Анастасии](https://github.com/AnastasiaMozhayskaya), [Романа](https://github.com/r-makushkin), Руслана.
#### Используемые Датасеты: [Соревнования из Kaggle](https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images), для Автоэнкодера использовался Датасет из GoogleDrive.

#### Список задач:
1. Создать модель автоэнкодера для очищения фотографий от шума📝.  
Подзадачи:

   - Создать архитектуру модели для кодирования и декодирования входного объекта.

   - Провести обучение, учитывая следующие нюансы: делать предсказание модели на объекте с шумом, уменьшать Loss-func на аналогичном объекте без шума.

   - Отобрать веса, которые показали наилучший результат на тестовой выборке.
![изображение](https://github.com/ruslan-DS/computervision_project/assets/146819015/e4bedcbc-aa02-478b-9055-d4b972cc26d8)


2. Создать модели для детекция судов и кораблей на фотоснимках🔎.  
Подзадачи:

   - Загрузить размеченный датасет для детекции.

   - Протестировать результаты детекции на нескольких моделях (в приоритете были модели YOLOv5 и YOLOv8).

   - Выбрать наиболее оптимальную модель (веса) по объему памяти, количеству слоев и скорости предсказания.
![изображение](https://github.com/ruslan-DS/computervision_project/assets/146819015/5c658bbd-c610-4496-8e75-b39e3f5bdfc7)

#### Вывод и результаты представлены на сайте во вкладе 🔥 Results.
