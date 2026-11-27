"""
Executa o pipeline completo do experimento em ordem reprodutível.

Uso:
    python run_all.py
    python run_all.py --skip-eda
    python run_all.py --skip-benchmark
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("preproc", "scripts/run_preprocessing.py", "1/7 Pré-processamento e engenharia de atributos"),
    ("eda", "scripts/generate_eda_figures.py", "2/7 Figuras da Análise Exploratória (EDA)"),
    ("rf", "scripts/train_random_forest.py", "3/7 Treinamento e avaliação do Random Forest"),
    ("xgb", "scripts/train_xgboost.py", "4/7 Otimização bayesiana e treinamento do XGBoost"),
    ("ext", "scripts/evaluate_external_validation.py", "5/7 Validação externa no JAC iEV40 (CAN)"),
    ("comp", "scripts/compare_intra_vs_external.py", "6/7 Comparação intra-dataset (Cenários 1 e 2) vs. externa (Cenário 3)"),
    ("bench", "scripts/benchmark_inference.py", "7/7 Benchmark de latência de inferência em CPU"),
]


def main():
    parser = argparse.ArgumentParser(description="Executa o pipeline completo de reprodução do TCC.")
    parser.add_argument("--skip-eda", action="store_true", help="Pula a geração das figuras de EDA.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Pula o benchmark de inferência.")
    args = parser.parse_args()

    t_total = time.time()
    for key, rel_path, label in STAGES:
        if key == "eda" and args.skip_eda:
            print(f"\n[PULADO] {label}")
            continue
        if key == "bench" and args.skip_benchmark:
            print(f"\n[PULADO] {label}")
            continue

        script = ROOT / rel_path
        print("\n" + "=" * 78)
        print(f"Executando: {label} ({rel_path})")
        print("=" * 78)
        t0 = time.time()
        result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
        if result.returncode != 0:
            print(f"\n[ERRO] Falha na etapa '{label}' (código {result.returncode}).")
            sys.exit(result.returncode)
        print(f"[OK] {label} concluída em {time.time() - t0:.1f}s")

    print("\n" + "=" * 78)
    print(f"Pipeline completo finalizado com sucesso em {time.time() - t_total:.1f}s!")
    print("Resultados salvos em: results/, figures/ e models/")
    print("=" * 78)


if __name__ == "__main__":
    main()
