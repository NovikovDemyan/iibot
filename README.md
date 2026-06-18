# CS2 v2 Next Tools v1

Следующий безопасный этап после обучения второй модели.

Пакет делает:
- сравнение v1/v2 по метрикам;
- проверку v2 на папке кадров;
- live-preview выбора цели;
- демо-видео, где видно, какую цель агент выбрал бы.

Пакет НЕ делает:
- не управляет мышью;
- не нажимает клавиши;
- не стреляет;
- не читает память CS2;
- не внедряется в процесс;
- не обходит античит.

## Куда положить

Положи файлы в папку проекта `cs2_vision_agent_safe_capture_v3`.

## 1. Установить зависимости

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_next.txt
```

## 2. Сравнить v1 и v2

```powershell
.\.venv\Scripts\python.exe compare_models.py --data dataset/yolo_enemy_v2/data.yaml --models "runs\detect\runs\detect\cs2_enemy_detector\weights\best.pt" "runs\cs2_enemy_detector_v2\weights\best.pt"
```

Если путь к v2 другой, замени его.

## 3. Проверить v2 на кадрах

```powershell
.\.venv\Scripts\python.exe predict_on_folder_v2.py --model "runs\cs2_enemy_detector_v2\weights\best.pt" --source dataset/raw_frames --out predictions_v2 --conf 0.25 --limit 500
```

## 4. Live-preview выбора цели

Только визуализация. Скрипт показывает:
- найденные коробки;
- выбранную цель;
- точку, куда агент смотрел бы;
- смещение от центра экрана.

```powershell
.\.venv\Scripts\python.exe live_target_preview.py --model "runs\cs2_enemy_detector_v2\weights\best.pt" --conf 0.25
```

Выход:
- `q` в окне;
- закрыть окно;
- `Ctrl+C` в PowerShell.

## 5. Демо-видео для диплома

```powershell
.\.venv\Scripts\python.exe make_target_demo_video.py --model "runs\cs2_enemy_detector_v2\weights\best.pt" --source dataset/raw_frames --out target_demo_v2.mp4 --max-frames 800 --conf 0.25
```

## Что смотреть

Если v2:
- `recall` вырос — хорошо, модель меньше пропускает врагов;
- `precision` сильно упал — много ложных срабатываний, надо добавить отрицательные кадры без врагов;
- `mAP50` выше — модель стала лучше в целом.
