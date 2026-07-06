# 🧪 Scients — UEBA Anomaly Detection in System Logs

**Transformer-based (BERT MLM)** система обнаружения аномалий в логах суперкомпьютера
Blue Gene/L (BGL) с экспериментальным сравнением против классических методов.

## 📋 Описание

Проект решает задачу **User and Entity Behavior Analytics (UEBA)** — обнаружение
аномального поведения в последовательностях системных событий. Модель обучается
на нормальных паттернах поведения и детектирует отклонения.

### Архитектура

```
[Логи BGL] → [Preprocessing] → [Session Encoding] → [Transformer (BERT MLM)]
                                                            ↓
                                                    [Anomaly Score]
                                                            ↓
                                              [Threshold (MCC + Sigma)]
                                                            ↓
                                               [Normal / Suspicious / Anomaly]
```

### Ключевые компоненты

| Компонент | Описание |
|-----------|----------|
| **Masked Language Modeling** | Обучение без учителя — модель предсказывает замаскированные события |
| **Center Loss** | Кластеризация нормальных сессий в пространстве эмбеддингов |
| **MCC Threshold Tuning** | Научно обоснованный подбор порога аномалии (Matthew's Correlation Coefficient) |
| **HPO (Random Search)** | Автоматический подбор архитектуры (embed_size, num_heads, num_layers) |
| **Baseline Comparison** | Isolation Forest + One-Class SVM для экспериментального доказательства |
| **LSTM Baseline (DeepLog)** | Нейросетевой бейзлайн — эмпирическое обоснование выбора Transformer |
| **Ablation Study** | Измерение вклада Center Loss / Stochastic Scoring / Max-агрегации |
| **Per-Category Analysis** | Detection rate по типам аномалий BGL (KERNDTLB, KERNSTOR, ...) |
| **Explainability** | Топ-N неожиданных событий аномальной сессии — объяснение для аналитика |
| **Multi-seed Protocol** | mean±std по сидам (results_log.csv + aggregate_results.py) |

## 🗂 Структура файлов

```
scients/
├── main.py              # Главный пайплайн (Train → Evaluate → Compare)
├── config.py            # Все гиперпараметры и настройки
├── seed.py              # Фиксация seeds для воспроизводимости
├── preprocessing.py     # Загрузка BGL, парсинг меток, кодирование
├── dataset.py           # MaskedLogDataset + MLM collate function
├── model.py             # TransformerPredictor (BERT-style)
├── loss.py              # Center Loss для One-Class кластеризации
├── trainer.py           # Обучение, пороги, HPO
├── evaluator.py         # Метрики (+PR-AUC) + per-category разбор + визуализация
├── predictor.py         # Inference на единичных сессиях (+ explainability отчёт)
├── baseline.py          # Isolation Forest + One-Class SVM
├── baseline_lstm.py     # LSTM next-event prediction (DeepLog-style) — нейросетевой бейзлайн
├── explainer.py         # Объяснимость: топ-N неожиданных событий аномальной сессии
├── run_ablation.py      # Ablation study: Center Loss / passes / max-vs-mean
├── aggregate_results.py # Multi-seed агрегация: mean±std из results_log.csv
├── utils.py             # Вспомогательные функции (сплиты, results log)
├── synthetic_data.py    # Генерация синтетических данных (fallback)
├── requirements.txt     # Зависимости
└── README.md            # Этот файл
```

## 🚀 Запуск

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Обучение + Оценка + Сравнение
```bash
# Полный пайплайн (реальные метки BGL)
python main.py

# Быстрый тест без HPO
python main.py --no_hpo --epochs 5

# Без baseline сравнения (быстрее)
python main.py --no_baselines

# Конкретный seed для воспроизводимости
python main.py --seed 42

# Без LSTM-бейзлайна
python main.py --no_lstm

# Ablation study (Center Loss / stochastic passes / max-vs-mean)
python run_ablation.py --epochs 15 --seed 42

# Multi-seed протокол (mean±std)
python main.py --seed 1 --tag multiseed --no_hpo
python main.py --seed 2 --tag multiseed --no_hpo
python main.py --seed 3 --tag multiseed --no_hpo
python aggregate_results.py --tag multiseed
```

### Google Colab
1. Загрузить `scients_package.zip` в Colab
2. Загрузить `BGL.log_structured.csv` (опционально — без него используются синтетические данные)
3. Открыть `scients_scientific_runner.ipynb` и выполнить все ячейки

## 📊 Метрики

| Метрика | Описание | Почему важна |
|---------|----------|-------------|
| **MCC** | Matthew's Correlation Coefficient | Устойчива к дисбалансу классов |
| **F1** | Гармоническое среднее Precision и Recall | Баланс точности и полноты |
| **F2** | F-beta с β=2 (акцент на Recall) | Критично для security — не пропускать атаки |
| **ROC-AUC** | Area Under ROC Curve | Общая дискриминантная способность |
| **PR-AUC** | Average Precision | При ~7.5% аномалий информативнее ROC-AUC |
| **Precision** | TP / (TP + FP) | Минимизация ложных тревог |
| **Recall** | TP / (TP + FN) | Обнаружение всех аномалий |

## 📚 Датасет

**BGL (Blue Gene/L)** — логи суперкомпьютера из Lawrence Livermore National Laboratory.
- **4,747,963** лог-записей
- **~7.5%** реальных аномалий (помечены вручную)
- Категории: KERNDTLB, KERNSTOR, KERNMNTF, KERNTERM, и др.

Ссылка: [LogHub](https://github.com/logpai/loghub)

## 📖 Научная основа

- **Attention Is All You Need** (Vaswani et al., 2017) — архитектура Transformer
- **BERT** (Devlin et al., 2018) — Masked Language Modeling
- **LogBERT** (Guo et al., 2021) — BERT для обнаружения аномалий в логах
- **DeepLog** (Du et al., 2017) — LSTM next-event prediction (реализован как нейросетевой бейзлайн)
- **Center Loss** (Wen et al., 2016) — кластеризация в пространстве признаков
- **MCC** (Matthews, 1975) — метрика для несбалансированных данных
