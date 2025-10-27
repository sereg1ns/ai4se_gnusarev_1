# HW 1

# Перед работой
- python -m pip install poetry
- poetry config virtualenvs.in-project true
- poetry install

# Запуск предобработки данных
poetry run python src/ai4se_gnusarev_1/internal/dataset/main.py --filters remove_url,expand_contraction,remove_special_chars,remove_repetitions,replace_curse_rephrasing --lower true --input <dataset_path> --output <output_path>

# Запуск обучения модели
poetry run python src/ai4se_gnusarev_1/main.py --model_type <model_type> --cfg_path <path_to_model_config>
Конфигурация модели содержит пути к данным, параметры модели, параметры обучения и т.п.

# Подбор параметров для RandomForest
poetry run python src/ai4se_gnusarev_1/internal/models/random_forest/search_params.py --cfg_path <path_to_parameter_search>
В конфигурации подбора параметров указываются путь к данным, статичные параметры и те параметры, которые мы хотим варьировать
