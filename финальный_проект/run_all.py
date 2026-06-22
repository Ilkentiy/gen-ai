import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_command(cmd: str, description: str):
    print(f"\n{description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and "Event loop is closed" not in result.stderr:
        print("Ошибки:")
        print(result.stderr)
    if result.returncode != 0:
        print(f"Ошибка: {description}")
        return False
    print(f"{description} - готово")
    return True

def main():
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    Path("input/kb").mkdir(exist_ok=True)
    
    if not run_command("python prepare_data.py", "Подготовка данных"):
        return
    
    if not run_command("python pipeline.py --limit 100", "Обработка тикетов (100 шт)"):
        return
    
    if not run_command("python eval.py", "Оценка на gold-наборе"):
        return
    
    print("\nРезультаты в папке output/")
    print("  - predictions.json")
    print("  - eval_results.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        if "Event loop is closed" not in str(e):
            print(f"Ошибка: {e}")