from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import products_to_names, split_products
from src.vlm_recognizer import recognize_products

METRIC_COLUMNS = ["precision", "recall", "f1_score"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
}
COMPARISON_COLUMNS = [
    "image_file",
    "manual_expected_count",
    "model_predicted_count",
    "matched_count",
    "missed_by_model_count",
    "invented_by_model_count",
    "manual_expected_products",
    "model_predicted_products",
    "matched_products",
    "missed_by_model_products",
    "invented_by_model_products",
    *METRIC_COLUMNS,
]


def count_products(value: object) -> int:
    if pd.isna(value):
        return 0
    return len(split_products(str(value)))


def evaluate_row(image_path: Path, expected_products: str) -> dict:
    recognition = recognize_products(image_path)
    predicted_names = products_to_names(recognition.get("products", []))

    expected = split_products(expected_products)
    predicted = split_products(", ".join(predicted_names))

    found = expected & predicted
    missed = expected - predicted
    invented = predicted - expected

    precision = len(found) / len(predicted) if predicted else 0.0
    recall = len(found) / len(expected) if expected else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "image_file": image_path.name,
        "manual_expected_count": len(expected),
        "model_predicted_count": len(predicted),
        "matched_count": len(found),
        "missed_by_model_count": len(missed),
        "invented_by_model_count": len(invented),
        "expected_products": ", ".join(sorted(expected)),
        "predicted_products": ", ".join(sorted(predicted)),
        "found": ", ".join(sorted(found)),
        "missed_by_model": ", ".join(sorted(missed)),
        "invented_by_model": ", ".join(sorted(invented)),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
        "summary": recognition.get("summary", ""),
    }


def ensure_comparison_columns(result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()
    if "manual_expected_count" not in result:
        result["manual_expected_count"] = result["expected_products"].apply(count_products)
    if "model_predicted_count" not in result:
        result["model_predicted_count"] = result["predicted_products"].apply(count_products)
    if "matched_count" not in result:
        result["matched_count"] = result["found"].apply(count_products)
    if "missed_by_model_count" not in result:
        result["missed_by_model_count"] = result["missed_by_model"].apply(count_products)
    if "invented_by_model_count" not in result:
        result["invented_by_model_count"] = result["invented_by_model"].apply(count_products)
    return result


def build_comparison_table(result: pd.DataFrame) -> pd.DataFrame:
    """Собирает таблицу сравнения ручной разметки и ответа модели."""
    result = ensure_comparison_columns(result)
    comparison = pd.DataFrame(
        {
            "image_file": result["image_file"],
            "manual_expected_count": result["manual_expected_count"],
            "model_predicted_count": result["model_predicted_count"],
            "matched_count": result["matched_count"],
            "missed_by_model_count": result["missed_by_model_count"],
            "invented_by_model_count": result["invented_by_model_count"],
            "manual_expected_products": result["expected_products"],
            "model_predicted_products": result["predicted_products"],
            "matched_products": result["found"],
            "missed_by_model_products": result["missed_by_model"],
            "invented_by_model_products": result["invented_by_model"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1_score"],
        }
    )
    return comparison[COMPARISON_COLUMNS]


def build_metrics_summary(result: pd.DataFrame) -> pd.DataFrame:
    """Собирает итоговую таблицу по всем основным метрикам Eval."""
    summary = result[METRIC_COLUMNS].mean().reset_index()
    summary.columns = ["metric", "mean_value"]
    summary["metric"] = summary["metric"].map(METRIC_LABELS)
    summary["mean_value"] = summary["mean_value"].round(3)
    return summary


def save_comparison_plots(comparison: pd.DataFrame, out_path: Path) -> tuple[Path, Path]:
    base_path = out_path.with_suffix("")
    expected_vs_model_path = base_path.with_name(f"{base_path.name}_manual_vs_model.png")
    match_errors_path = base_path.with_name(f"{base_path.name}_match_errors.png")

    counts_df = comparison[
        ["image_file", "manual_expected_count", "model_predicted_count"]
    ].rename(
        columns={
            "manual_expected_count": "Ручная разметка",
            "model_predicted_count": "Модель",
        }
    )
    counts_df.plot(
        x="image_file",
        y=["Ручная разметка", "Модель"],
        kind="bar",
        figsize=(12, 6),
        rot=45,
        title="Сравнение количества продуктов: ручная разметка и модель",
    )
    plt.xlabel("Изображение")
    plt.ylabel("Количество продуктов")
    plt.legend(title="Источник")
    plt.tight_layout()
    plt.savefig(expected_vs_model_path, dpi=160)
    plt.close()

    errors_df = comparison[
        ["image_file", "matched_count", "missed_by_model_count", "invented_by_model_count"]
    ].rename(
        columns={
            "matched_count": "Совпало",
            "missed_by_model_count": "Пропущено",
            "invented_by_model_count": "Лишнее",
        }
    )
    errors_df.plot(
        x="image_file",
        y=["Совпало", "Пропущено", "Лишнее"],
        kind="bar",
        figsize=(12, 6),
        rot=45,
        title="Сравнение совпадений, пропусков и лишних продуктов",
    )
    plt.xlabel("Изображение")
    plt.ylabel("Количество продуктов")
    plt.legend(title="Результат сравнения")
    plt.tight_layout()
    plt.savefig(match_errors_path, dpi=160)
    plt.close()

    return expected_vs_model_path, match_errors_path


def save_eval_curves_plot(result: pd.DataFrame, comparison: pd.DataFrame, out_path: Path) -> Path:
    """Сохраняет общий график в стиле учебных графиков Loss/Accuracy."""
    base_path = out_path.with_suffix("")
    curves_path = base_path.with_name(f"{base_path.name}_eval_curves.png")

    x_values = list(range(1, len(result) + 1))
    x_labels = comparison["image_file"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111820")
    fig.suptitle("Графики оценки распознавания", color="white", fontsize=18, fontweight="bold", x=0.02, ha="left")

    errors_ax, metrics_ax = axes

    errors_ax.set_facecolor("white")
    errors_ax.plot(x_values, comparison["missed_by_model_count"], marker="o", label="Пропущено моделью")
    errors_ax.plot(x_values, comparison["invented_by_model_count"], marker="o", label="Лишнее от модели")
    errors_ax.set_title("Errors")
    errors_ax.set_xlabel("Image")
    errors_ax.set_ylabel("Count")
    errors_ax.set_xticks(x_values)
    errors_ax.set_xticklabels(x_labels, rotation=45, ha="right")
    errors_ax.grid(True, alpha=0.6)
    errors_ax.legend()

    metrics_ax.set_facecolor("white")
    metrics_ax.plot(x_values, result["precision"], marker="o", label="Precision")
    metrics_ax.plot(x_values, result["recall"], marker="o", label="Recall")
    metrics_ax.plot(x_values, result["f1_score"], marker="o", label="F1-score")
    metrics_ax.set_title("Metrics")
    metrics_ax.set_xlabel("Image")
    metrics_ax.set_ylabel("Score")
    metrics_ax.set_ylim(0, 1)
    metrics_ax.set_xticks(x_values)
    metrics_ax.set_xticklabels(x_labels, rotation=45, ha="right")
    metrics_ax.grid(True, alpha=0.6)
    metrics_ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(curves_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    return curves_path


def save_metrics_plots(result: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> tuple[Path, Path]:
    base_path = out_path.with_suffix("")
    by_image_plot_path = base_path.with_name(f"{base_path.name}_metrics_by_image.png")
    summary_plot_path = base_path.with_name(f"{base_path.name}_metrics_summary.png")

    plot_df = result[["image_file", *METRIC_COLUMNS]].copy()
    plot_df = plot_df.rename(columns=METRIC_LABELS)
    plot_df.plot(
        x="image_file",
        y=list(METRIC_LABELS.values()),
        kind="bar",
        figsize=(12, 6),
        ylim=(0, 1),
        rot=45,
        title="Метрики по каждому изображению",
    )
    plt.xlabel("Изображение")
    plt.ylabel("Значение метрики")
    plt.legend(title="Метрика")
    plt.tight_layout()
    plt.savefig(by_image_plot_path, dpi=160)
    plt.close()

    summary.plot(
        x="metric",
        y="mean_value",
        kind="bar",
        figsize=(7, 5),
        ylim=(0, 1),
        rot=0,
        legend=False,
        title="Средние метрики распознавания",
    )
    plt.xlabel("Метрика")
    plt.ylabel("Среднее значение")
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=160)
    plt.close()

    return by_image_plot_path, summary_plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ручная оценка качества распознавания продуктов")
    parser.add_argument("--expected", default="data/eval_expected.csv", help="CSV с колонками image_file, expected_products")
    parser.add_argument("--images-dir", default="data/eval_images", help="Папка с 10–20 фото для оценки")
    parser.add_argument("--out", default="outputs/eval_results.csv", help="Куда сохранить результаты")
    args = parser.parse_args()

    expected_path = Path(args.expected)
    images_dir = Path(args.images_dir)
    out_path = Path(args.out)

    if not expected_path.exists():
        raise FileNotFoundError(
            f"Не найден файл {expected_path}. Заполните data/eval_expected.csv по шаблону."
        )

    df = pd.read_csv(expected_path)
    required = {"image_file", "expected_products"}
    if not required.issubset(df.columns):
        raise ValueError("CSV должен содержать колонки: image_file, expected_products")

    rows = []
    for _, row in df.iterrows():
        image_path = images_dir / str(row["image_file"])
        if not image_path.exists():
            print(f"Пропуск: не найден файл {image_path}")
            continue
        print(f"Оцениваю: {image_path}")
        rows.append(evaluate_row(image_path, str(row["expected_products"])))

    result = pd.DataFrame(rows)
    result = ensure_comparison_columns(result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    if len(result) > 0:
        comparison = build_comparison_table(result)
        comparison_path = out_path.with_suffix("").with_name(f"{out_path.stem}_manual_vs_model.csv")
        comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        expected_vs_model_path, match_errors_path = save_comparison_plots(comparison, out_path)
        eval_curves_path = save_eval_curves_plot(result, comparison, out_path)

        summary = build_metrics_summary(result)
        summary_path = out_path.with_suffix("").with_name(f"{out_path.stem}_metrics_summary.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        by_image_plot_path, summary_plot_path = save_metrics_plots(result, summary, out_path)

        print("\n=== Итоговая оценка ===")
        print(f"Количество фото: {len(result)}")
        print("\nСравнение ручной разметки и ответа модели:")
        print(
            comparison[
                [
                    "image_file",
                    "manual_expected_count",
                    "model_predicted_count",
                    "matched_count",
                    "missed_by_model_count",
                    "invented_by_model_count",
                ]
            ].to_string(index=False)
        )
        print("\nТаблица средних метрик:")
        print(summary.to_string(index=False))
        print(f"\nТаблица сравнения сохранена: {comparison_path}")
        print(f"График ручной разметки и модели сохранён: {expected_vs_model_path}")
        print(f"График совпадений и ошибок сохранён: {match_errors_path}")
        print(f"Общий график оценки сохранён: {eval_curves_path}")
        print(f"\nТаблица метрик сохранена: {summary_path}")
        print(f"График метрик по фото сохранён: {by_image_plot_path}")
        print(f"График средних метрик сохранён: {summary_plot_path}")
    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
