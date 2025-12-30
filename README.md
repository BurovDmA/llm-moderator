# llm-moderator

Модель для автоматического определения нарушений правил на основе текста и связанного с ним контекста (правила и примеры).
Проект является реализацией MLOps-пайплайна с использованием PyTorch Lightning, Hydra, DVC, MLflow и кастомного CLI.

---

# Содержание

- [Описание проекта](#описание-проекта)
- [Установка и настройка окружения](#установка-и-настройка-окружения)
- [Работа с данными (DVC)](#работа-с-данными-dvc)
- [Конфигурации (Hydra)](#конфигурации-hydra)
- [Запуск обучения](#запуск-обучения)
- [CLI команды](#cli-команды)

---

# Описание проекта

Цель проекта — построить модель модерации текста, которая по входному тексту, правилу и примерам классифицирует, нарушает ли текст правило.

В качестве модели используется `distilbert-base-uncased`, обучаемый как классификатор с 2 классами (`0` — нет нарушения, `1` — нарушение).
Данные взяты из набора **Jigsaw Agile Community Rules**, где каждому тексту сопоставлено правило и примеры.

Пайплайн:

1. Загрузка данных с HuggingFace.
2. Препроцессинг: генерация промпта (`text`) на основе колонок `body + rule + examples`.
3. Деление на train/val.
4. Трекинг данных через DVC.
5. Обучение Lightning-модели.
6. Логирование в MLflow.
7. Настройки через Hydra.
8. CLI для удобного запуска.

---

# ⚙ Установка и настройка окружения (Setup раздел)

## 1. Клонирование репозитория

```bash
git clone https://github.com/<your-username>/llm-moderator.git
cd llm-moderator

2. Создание виртуального окружения для проверки ДЗ (Если вдруг не создано еще), пример через conda:
conda create --name llm_validator python=3.11
conda activate llm_validator

3. Установка poetry

Если poetry не установлен:

pip install poetry

4. Установка зависимостей
poetry install

4. Установка зависимостей, которые poetry не может добавить (Иногда ловил баг на Windows, на unix все работает исправно)
poetry run pip install mlflow fire

6. Установка pre-commit
pre-commit install
pre-commit run -a

Все проверки должны быть зелёными.

7. Запуск обучения (train раздел):

1. Запуск MLflow UI (в отдельном терминале)
mlflow ui --host 127.0.0.1 --port 8080

2. Запуск обучения
poetry run llm-cli train - вообще предварительно можно скачать данные через poetry run llm-cli download, но "под капотом" train ее сам запустит,
выдав лог-error о том что данных нет, так что в целом проверяющему для удобства достаточно только этой команды, она все сделает)


Работа с данными (DVC)

Проект использует DVC для версиирования данных.
Первичное скачивание данных выполняется командой:

poetry run llm-cli download

Если у вас данных нет — при запуске обучения автоматически выполнится:
dvc pull

Данные хранятся в локальном DVC-remote (dvc_storage/).

🔧 Конфигурации (Hydra)

Все конфиги находятся в llm_moderator/configs/.

Примеры:

configs/data/default.yaml — параметры данных

configs/model/default.yaml — параметры модели

configs/train/default.yaml — обучение (lr, epochs, device)

configs/mlflow/default.yaml — трекинг в MLflow

config.yaml — главный конфиг

Переключить параметр можно так:

poetry run llm-cli train train.max_epochs=1 data.batch_size=4

Запуск обучения
1. Запуск MLflow UI (в отдельном терминале)
mlflow ui --host 127.0.0.1 --port 8080

2. Запуск обучения
poetry run llm-cli train


Что происходит:

DVC подтягивает data/processed

Hydra собирает конфиг

Lightning запускает обучение

MLflow логирует:

train/val loss

train/val accuracy

train/val ROC-AUC

гиперпараметры

git commit hash

После запуска зайдите в MLflow:

http://127.0.0.1:8080

CLI команды

Все команды доступны через:

poetry run llm-cli <команда>


Доступные команды:

Скачать данные
poetry run llm-cli download

Запустить обучение
poetry run llm-cli train

