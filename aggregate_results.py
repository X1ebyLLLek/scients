"""
aggregate_results.py — Агрегация метрик по нескольким сидам (mean ± std).

Multi-seed протокол: main.py после каждого запуска дописывает финальные метрики
в results_log.csv. Этот скрипт группирует записи по тегу эксперимента и печатает
mean ± std — оценку устойчивости результатов, необходимую для научной работы
(одиночный прогон не позволяет судить о разбросе).

Запуск нескольких сидов (Colab):
  for seed in 1 2 3: python main.py --seed $seed --tag multiseed --no_hpo
Затем:
  python aggregate_results.py
"""

import argparse

import pandas as pd

from config import Config

METRICS = ['mcc', 'f1', 'f2', 'precision', 'recall', 'roc_auc', 'pr_auc']


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument("--log", type=str, default=Config.RESULTS_LOG_PATH)
    parser.add_argument("--tag", type=str, default=None, help="Filter by experiment tag")
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    if args.tag:
        df = df[df['tag'] == args.tag]

    if df.empty:
        print(f"No records found in {args.log}" + (f" for tag '{args.tag}'" if args.tag else ""))
        return

    print(f"Loaded {len(df)} runs from {args.log}")
    print()

    for tag, group in df.groupby('tag'):
        seeds = sorted(group['seed'].unique())
        print(f"=== Tag: {tag} ({len(group)} runs, seeds={seeds}) ===")
        print(f"{'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 56)
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = group[metric].astype(float)
            print(f"{metric:<12} {values.mean():>10.4f} {values.std():>10.4f} "
                  f"{values.min():>10.4f} {values.max():>10.4f}")
        print()

        # Строка для диплома: MCC = 0.83 ± 0.01
        summary = ", ".join(
            f"{m.upper()}={group[m].mean():.3f}±{group[m].std():.3f}"
            for m in METRICS if m in group.columns
        )
        print(f"Для текста: {summary}")
        print()


if __name__ == "__main__":
    main()
