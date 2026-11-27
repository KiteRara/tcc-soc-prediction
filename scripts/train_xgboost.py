"""
Treinamento, Otimização Bayesiana e Avaliação do modelo XGBoost conforme Seção 4.8.2 do TCC.
Utiliza Optuna para busca bayesiana de hiperparâmetros, early stopping no conjunto de validação,
treina os modelos FULL e COMPAT, compara diretamente com o Random Forest (com IC 95% bootstrap)
e plota curvas de predição temporal conjuntas e análise de resíduos.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor

from src.common import (
    DATA_PROCESSED, FIGURES, MODELS, RESULTS, LABELS, TARGET,
    FEATURES_FULL, FEATURES_COMPAT, SEED, metrics, bootstrap_ci,
    save_json, load_json, set_style, savefig
)

# Silenciar logs excessivos do optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run():
    set_style()
    print("--- 1. Carregando dados e scalers ---")
    train_df = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    val_df = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    test_df = pd.read_parquet(DATA_PROCESSED / "test.parquet")

    scaler_full = joblib.load(MODELS / "scaler_full.joblib")
    scaler_compat = joblib.load(MODELS / "scaler_compat.joblib")

    X_train_full = scaler_full.transform(train_df[FEATURES_FULL])
    X_val_full = scaler_full.transform(val_df[FEATURES_FULL])
    X_test_full = scaler_full.transform(test_df[FEATURES_FULL])

    X_train_compat = scaler_compat.transform(train_df[FEATURES_COMPAT])
    X_val_compat = scaler_compat.transform(val_df[FEATURES_COMPAT])
    X_test_compat = scaler_compat.transform(test_df[FEATURES_COMPAT])

    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    # ----------------------------------------------------------------------------------
    # 2. Otimização Bayesiana de Hiperparâmetros com Optuna (Seção 4.8.2)
    # ----------------------------------------------------------------------------------
    print("\n--- 2. Otimização Bayesiana com Optuna e Early Stopping ---")
    # Subamostra para 15 trials rápidos e representativos
    sample_tune_idx = np.linspace(0, len(X_train_full) - 1, 60000, dtype=int)
    X_tune = X_train_full[sample_tune_idx]
    y_tune = y_train[sample_tune_idx]

    def objective(trial):
        params = {
            "n_estimators": 200,
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "random_state": SEED,
            "n_jobs": -1,
            "tree_method": "hist",
            "eval_metric": "rmse",
            "early_stopping_rounds": 20
        }
        model = XGBRegressor(**params)
        model.fit(
            X_tune, y_tune,
            eval_set=[(X_val_full, y_val)],
            verbose=False
        )
        preds_val = model.predict(X_val_full)
        return float(np.sqrt(np.mean((preds_val - y_val)**2)))

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=15, timeout=180)
    opt_time = time.time() - t0
    best_params = study.best_params
    print(f"Otimização Bayesiana concluída em {opt_time:.1f}s (15 trials).")
    print(f"Melhores hiperparâmetros: {best_params}")
    print(f"Melhor RMSE na validação durante a busca: {study.best_value:.3f}%")

    # ----------------------------------------------------------------------------------
    # 3. Treinamento do Modelo Final FULL com Early Stopping no Treino Completo
    # ----------------------------------------------------------------------------------
    print("\n--- 3. Treinando XGBoost FULL no treino completo ---")
    final_params = {
        **best_params,
        "n_estimators": 500,
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "early_stopping_rounds": 25
    }
    xgb_full = XGBRegressor(**final_params)
    t0 = time.time()
    xgb_full.fit(
        X_train_full, y_train,
        eval_set=[(X_train_full, y_train), (X_val_full, y_val)],
        verbose=False
    )
    train_time_full = time.time() - t0
    best_iteration = xgb_full.best_iteration
    print(f"XGBoost FULL treinado em {train_time_full:.1f}s (Parada antecipada na árvore {best_iteration})")
    joblib.dump(xgb_full, MODELS / "xgb_full.joblib")

    # Predições FULL
    p_train_full = xgb_full.predict(X_train_full)
    p_val_full = xgb_full.predict(X_val_full)
    p_test_full = xgb_full.predict(X_test_full)

    m_train_full = metrics(y_train, p_train_full)
    m_val_full = metrics(y_val, p_val_full)
    m_test_full = metrics(y_test, p_test_full)
    ci_test_full = bootstrap_ci(y_test, p_test_full, test_df["trip_id"])

    print("\nResultados XGBoost (FULL):")
    print(f"  Treino:    RMSE={m_train_full['RMSE']:.3f} | MAE={m_train_full['MAE']:.3f} | R²={m_train_full['R2']:.4f}")
    print(f"  Validação: RMSE={m_val_full['RMSE']:.3f} | MAE={m_val_full['MAE']:.3f} | R²={m_val_full['R2']:.4f}")
    print(f"  Teste:     RMSE={m_test_full['RMSE']:.3f} | MAE={m_test_full['MAE']:.3f} | R²={m_test_full['R2']:.4f}")
    print(f"  IC 95% Teste RMSE: {ci_test_full['RMSE_ci']}, MAE: {ci_test_full['MAE_ci']}")

    # ----------------------------------------------------------------------------------
    # 4. Treinamento do Modelo COMPAT (para validação externa com JAC iEV40)
    # ----------------------------------------------------------------------------------
    print("\n--- 4. Treinando XGBoost COMPAT ---")
    xgb_compat = XGBRegressor(**final_params)
    t0 = time.time()
    xgb_compat.fit(
        X_train_compat, y_train,
        eval_set=[(X_val_compat, y_val)],
        verbose=False
    )
    train_time_compat = time.time() - t0
    print(f"XGBoost COMPAT treinado em {train_time_compat:.1f}s (Parada antecipada na árvore {xgb_compat.best_iteration})")
    joblib.dump(xgb_compat, MODELS / "xgb_compat.joblib")

    p_test_compat = xgb_compat.predict(X_test_compat)
    m_test_compat = metrics(y_test, p_test_compat)
    print(f"  Teste COMPAT: RMSE={m_test_compat['RMSE']:.3f} | MAE={m_test_compat['MAE']:.3f} | R²={m_test_compat['R2']:.4f}")

    # ----------------------------------------------------------------------------------
    # 5. Feature Importance do XGBoost
    # ----------------------------------------------------------------------------------
    print("\n--- 5. Plotando Feature Importance do XGBoost ---")
    gain_dict = xgb_full.get_booster().get_score(importance_type="gain")
    # mapear f0..fn para nomes originais
    f_map = {f"f{i}": col for i, col in enumerate(FEATURES_FULL)}
    gain_series = pd.Series({f_map.get(k, k): v for k, v in gain_dict.items()}).sort_values(ascending=False)
    gain_df = pd.DataFrame({
        "feature": gain_series.index,
        "gain": gain_series.values,
        "label": [LABELS.get(f, f) for f in gain_series.index]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="gain", y="label", data=gain_df, palette="flare_r", ax=ax, hue="label", legend=False)
    ax.set_title("Importância de Atributos (Gain) — XGBoost Regressor", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ganho Médio nas Divisões (Average Gain)")
    ax.set_ylabel("Atributo de Telemetria")
    plt.tight_layout()
    savefig(fig, "fig_xgb_feature_importance")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 6. Resíduos e Scatter Real vs Predito do XGBoost
    # ----------------------------------------------------------------------------------
    print("--- 6. Gerando gráficos de resíduos do XGBoost ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sample_sub = test_df.sample(n=min(8000, len(test_df)), random_state=SEED)
    sub_idx = sample_sub.index
    y_sub = y_test[sub_idx]
    p_sub = p_test_full[sub_idx]
    residuals = p_sub - y_sub

    # Scatter Real vs Predito
    ax1.scatter(y_sub, p_sub, alpha=0.3, s=10, color="#C62828")
    lims = [0, 100]
    ax1.plot(lims, lims, "k--", linewidth=1.5, label="Predição Ideal (y = ŷ)")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title(f"XGBoost: Valor Real vs Predito (Teste)\nRMSE: {m_test_full['RMSE']:.2f}% | R²: {m_test_full['R2']:.4f}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("SOC Real [%]")
    ax1.set_ylabel("SOC Predito [%]")
    ax1.legend(frameon=True)

    # Distribuição dos Resíduos
    sns.histplot(residuals, bins=40, kde=True, color="#C62828", ax=ax2, alpha=0.6)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax2.axvline(np.mean(residuals), color="blue", linestyle=":", linewidth=1.5, label=f"Viés Médio: {np.mean(residuals):.2f}%")
    ax2.set_title("Distribuição dos Resíduos do XGBoost no Teste", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Erro Residual (SOC Predito − SOC Real) [%]")
    ax2.set_ylabel("Frequência")
    ax2.legend(frameon=True)
    plt.tight_layout()
    savefig(fig, "fig_xgb_residuals_and_scatter")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 7. Comparação Direta RF vs XGBoost: Curvas Temporais de Viagens do Teste
    # ----------------------------------------------------------------------------------
    print("--- 7. Gerando perfis temporais comparativos (Real vs RF vs XGB) ---")
    rf_full = joblib.load(MODELS / "rf_full.joblib")
    test_trips = test_df.groupby("trip_id").filter(lambda x: len(x) > 150 and (x["soc"].max() - x["soc"].min() > 2))
    sample_trips = test_trips["trip_id"].unique()[:4]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    for ax, tid in zip(axes.flat, sample_trips):
        t_mask = (test_df["trip_id"] == tid).to_numpy()
        sub_t = test_df[t_mask].sort_values("ts_ms").reset_index(drop=True)
        time_min = (sub_t["ts_ms"] - sub_t["ts_ms"].iloc[0]) / 60000.0
        
        y_real = sub_t["soc"].to_numpy()
        X_t = scaler_full.transform(sub_t[FEATURES_FULL])
        y_rf = rf_full.predict(X_t)
        y_xgb = xgb_full.predict(X_t)

        rmse_rf = np.sqrt(np.mean((y_rf - y_real)**2))
        rmse_xgb = np.sqrt(np.mean((y_xgb - y_real)**2))

        ax.plot(time_min, y_real, color="#263238", linewidth=2.8, label="SOC Real [%]")
        ax.plot(time_min, y_rf, color="#2E7D32", linestyle="--", linewidth=1.8, label=f"Random Forest (RMSE: {rmse_rf:.2f}%)")
        ax.plot(time_min, y_xgb, color="#C62828", linestyle="-.", linewidth=1.8, label=f"XGBoost (RMSE: {rmse_xgb:.2f}%)")
        
        ax.set_title(f"Viagem ID: {tid} (Veículo {int(sub_t['veh'].iloc[0])})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tempo decorrido [minutos]")
        ax.set_ylabel("SOC [%]")
        ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    savefig(fig, "fig_comparison_temporal_trips")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 8. Gráfico de Barras Comparativo das Métricas com Intervalo de Confiança 95%
    # ----------------------------------------------------------------------------------
    print("--- 8. Gerando gráfico de barras comparativo (RF vs XGB) ---")
    rf_meta = load_json("rf_metrics.json")
    rf_test = rf_meta["metrics_full"]["test"]
    rf_ci = rf_meta["metrics_full"]["test_ci"]

    comp_data = [
        {"Modelo": "Random Forest", "Métrica": "RMSE [%]", "Valor": rf_test["RMSE"],
         "ci_lo": rf_ci["RMSE_ci"][0], "ci_hi": rf_ci["RMSE_ci"][1]},
        {"Modelo": "XGBoost", "Métrica": "RMSE [%]", "Valor": m_test_full["RMSE"],
         "ci_lo": ci_test_full["RMSE_ci"][0], "ci_hi": ci_test_full["RMSE_ci"][1]},
        {"Modelo": "Random Forest", "Métrica": "MAE [%]", "Valor": rf_test["MAE"],
         "ci_lo": rf_ci["MAE_ci"][0], "ci_hi": rf_ci["MAE_ci"][1]},
        {"Modelo": "XGBoost", "Métrica": "MAE [%]", "Valor": m_test_full["MAE"],
         "ci_lo": ci_test_full["MAE_ci"][0], "ci_hi": ci_test_full["MAE_ci"][1]}
    ]
    comp_df = pd.DataFrame(comp_data)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_positions = np.array([0, 1, 3, 4])
    colors = ["#2E7D32", "#C62828", "#2E7D32", "#C62828"]
    labels = ["RF\n(RMSE)", "XGBoost\n(RMSE)", "RF\n(MAE)", "XGBoost\n(MAE)"]

    bars = ax.bar(x_positions, comp_df["Valor"], yerr=[comp_df["Valor"] - comp_df["ci_lo"], comp_df["ci_hi"] - comp_df["Valor"]],
                  capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.7)
    
    for bar, val in zip(bars, comp_df["Valor"]):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.35, f"{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Erro de Estimação [% de SOC]", fontsize=11)
    ax.set_title("Comparação de Desempenho no Conjunto de Teste (com IC 95% Bootstrap)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(comp_df["ci_hi"]) + 2.0)
    plt.tight_layout()
    savefig(fig, "fig_comparison_metrics_barchart")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 9. Salvando Resultados Consolidados
    # ----------------------------------------------------------------------------------
    xgb_results = {
        "best_params": best_params,
        "best_iteration": int(best_iteration),
        "train_time_full_s": train_time_full,
        "train_time_compat_s": train_time_compat,
        "metrics_full": {
            "train": m_train_full,
            "val": m_val_full,
            "test": m_test_full,
            "test_ci": ci_test_full
        },
        "metrics_compat": {
            "test": m_test_compat
        },
        "feature_importances": gain_df.to_dict(orient="records")
    }
    save_json(xgb_results, "xgb_metrics.json")

    comparison_summary = {
        "RandomForest": {
            "RMSE": rf_test["RMSE"], "RMSE_CI": rf_ci["RMSE_ci"],
            "MAE": rf_test["MAE"], "MAE_CI": rf_ci["MAE_ci"],
            "R2": rf_test["R2"],
            "compat_RMSE": rf_meta["metrics_compat"]["test"]["RMSE"],
            "compat_MAE": rf_meta["metrics_compat"]["test"]["MAE"],
            "compat_R2": rf_meta["metrics_compat"]["test"]["R2"]
        },
        "XGBoost": {
            "RMSE": m_test_full["RMSE"], "RMSE_CI": ci_test_full["RMSE_ci"],
            "MAE": m_test_full["MAE"], "MAE_CI": ci_test_full["MAE_ci"],
            "R2": m_test_full["R2"],
            "compat_RMSE": m_test_compat["RMSE"],
            "compat_MAE": m_test_compat["MAE"],
            "compat_R2": m_test_compat["R2"]
        }
    }
    save_json(comparison_summary, "model_comparison.json")
    print("\nTreinamento e Comparação do XGBoost finalizados com sucesso!")

if __name__ == "__main__":
    run()
