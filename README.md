## Computer Vision contest #1
Решение представляет собой baseline с непредобученой моделью (см. model.py).
Т.к. предсказываемые точки линейно зависимы - в предпоследнем линейном слое
ищем ключевые точки (число берем все равно с запасом), в последнем линейном уже ищем решение.
Первые слои более менее стандартные.

![model](https://github.com/OlegPozovnoy/MailRuCVContest1/blob/master/model.JPG)    
![result](https://github.com/OlegPozovnoy/MailRuCVContest1/blob/master/result.JPG)
