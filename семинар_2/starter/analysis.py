import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load(path: str) -> pd.DataFrame:
    """Читаем CSV."""
    return pd.read_csv(path, encoding="utf-8")


def plot_hist_ages(df: pd.DataFrame, out: str = "ages.png"):
    """Гистограмма возраста."""
    plt.figure(figsize=(8, 4))
    plt.hist(df["age"], bins=12, color="#4A90D9", edgecolor="white")
    plt.xlabel("Возраст")
    plt.ylabel("Число заявок")
    plt.title(f"Распределение возраста ({len(df)} заявок)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  - {out}")


def plot_exp_by_speciality(df: pd.DataFrame, out: str = "exp_by_speciality.png"):
    """Стаж по специальностям."""
    if "years_of_experience" not in df.columns or "speciality" not in df.columns:
        return
    groups = df.groupby("speciality")["years_of_experience"].apply(list)
    plt.figure(figsize=(12, 5))
    positions = range(1, len(groups) + 1)
    plt.boxplot(list(groups.values), positions=list(positions), vert=True)
    plt.xticks(list(positions), list(groups.index), rotation=45, ha="right", fontsize=8)
    plt.ylabel("Стаж, лет")
    plt.title("Стаж × специальность")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  - {out}")


def cross_table(df: pd.DataFrame) -> pd.DataFrame:
    """Кросс-таблица город × специальность."""
    if "city" not in df.columns or "speciality" not in df.columns:
        return pd.DataFrame()
    ct = pd.crosstab(df["city"], df["speciality"])
    ct.to_csv("cross_table.csv", encoding="utf-8-sig")
    print(f"  - cross_table.csv")
    return ct


def write_report(df: pd.DataFrame, out: str = "report.md"):
    """Отчёт."""
    n = len(df)
    lines = [f"# Отчёт по {n} заявкам\n"]

    # Топ городов
    cities = df["city"].value_counts()
    top_city_pct = cities.iloc[0] / n * 100
    lines.append("## Города\n")
    lines.append(f"- Уникальных: {len(cities)}")
    lines.append(f"- Топ-1: **{cities.index[0]}** — {cities.iloc[0]} ({top_city_pct:.0f}%)")
    if top_city_pct > 40:
        lines.append("- Превышен порог 40%")
    lines.append("")

    # Топ специальностей
    spec = df["speciality"].value_counts()
    top_spec_pct = spec.iloc[0] / n * 100
    lines.append("## Специальности\n")
    lines.append(f"- Уникальных: {len(spec)}")
    lines.append(f"- Топ-1: **{spec.index[0]}** — {spec.iloc[0]} ({top_spec_pct:.0f}%)")
    if top_spec_pct > 35:
        lines.append("- Превышен порог 35%")
    lines.append("")

    # Дубликаты имён (а не полных ФИО)
    names_list = []
    for full_name in df["full_name"]:
        parts = full_name.strip().split()
        if len(parts) >= 2:
            names_list.append(parts[1])  # берём имя
        else:
            names_list.append(full_name)
    
    names = pd.Series(names_list).value_counts()
    dupes = names[names > 1]
    lines.append("## Имена\n")
    lines.append(f"- Уникальных имён: {len(names)} из {n}")
    if len(dupes):
        lines.append(f"- Повторяющиеся имена: {dict(dupes.head(5))}")
    else:
        lines.append("- Повторов нет")
    lines.append("")

    # Кросс-таблица
    ct = cross_table(df)
    if not ct.empty:
        lines.append("## Кросс-таблица город × специальность\n")
        lines.append("```")
        lines.append(ct.to_string())
        lines.append("```")
        lines.append("")

    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"  - {out}")


def main(path: str = "applications.csv"):

    df = load(path)

    print("\nСохранено:")
    plot_hist_ages(df)
    plot_exp_by_speciality(df)
    cross_table(df)
    write_report(df)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "applications.csv"
    main(path)