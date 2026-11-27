"""
Treinamento, Otimização e Avaliação do modelo Random Forest conforme Seção 4.8.1 do TCC.
Executa Grid Search com TimeSeriesSplit, treina os modelos FULL e COMPAT,
extrai importância de atributos, calcula métricas (RMSE, MAE, R²) com IC 95%,
e gera gráficos diagnósticos de resíduos e séries temporais.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.common import (
    DATA_PROCESSED, FIGURES, MODELS, RESULTS, LABELS, TARGET,
    FEATURES_FULL, FEATURES_COMPAT, SEED, metrics, bootstrap_ci,
    save_json, set_style, savefig
)

def run():
    set_style()
    print("--- 1. Carregando datasets processados ---")
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

    print(f"Treino: {X_train_full.shape}, Validação: {X_val_full.shape}, Teste: {X_test_full.shape}")

    # ----------------------------------------------------------------------------------
    # 2. Grid Search com TimeSeriesSplit para Hiperparâmetros (Seção 4.8.1)
    # ----------------------------------------------------------------------------------
    print("\n--- 2. Grid Search com TimeSeriesSplit ---")
    # Subamostragem temporal uniforme de 40.000 pontos para busca em grade eficiente
    sample_idx = np.linspace(0, len(X_train_full) - 1, 40000, dtype=int)
    X_tune = X_train_full[sample_idx]
    y_tune = y_train[sample_idx]

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [15, 25],
        "max_features": ["sqrt", 0.6],
        "min_samples_leaf": [2, 5]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    rf_base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    t0 = time.time()
    grid_search.fit(X_tune, y_tune)
    tune_time = time.time() - t0
    best_params = grid_search.best_params_
    print(f"Melhores hiperparâmetros ({tune_time:.1f}s): {best_params}")
    print(f"Melhor RMSE na validação cruzada temporal: {-grid_search.best_score_:.3f}")

    # ----------------------------------------------------------------------------------
    # 3. Treinamento do Modelo Final FULL com o conjunto completo de treino
    # ----------------------------------------------------------------------------------
    print("\n--- 3. Treinando Random Forest FULL no treino completo ---")
    rf_full = RandomForestRegressor(
        **best_params,
        random_state=SEED,
        n_jobs=-1
    )
    t0 = time.time()
    rf_full.fit(X_train_full, y_train)
    train_time_full = time.time() - t0
    print(f"Treinamento RF FULL concluído em {train_time_full:.1f}s")
    joblib.dump(rf_full, MODELS / "rf_full.joblib")

    # Predições FULL
    p_train_full = rf_full.predict(X_train_full)
    p_val_full = rf_full.predict(X_val_full)
    p_test_full = rf_full.predict(X_test_full)

    m_train_full = metrics(y_train, p_train_full)
    m_val_full = metrics(y_val, p_val_full)
    m_test_full = metrics(y_test, p_test_full)
    ci_test_full = bootstrap_ci(y_test, p_test_full, test_df["trip_id"])

    print("\nResultados Random Forest (FULL):")
    print(f"  Treino:    RMSE={m_train_full['RMSE']:.3f} | MAE={m_train_full['MAE']:.3f} | R²={m_train_full['R2']:.4f}")
    print(f"  Validação: RMSE={m_val_full['RMSE']:.3f} | MAE={m_val_full['MAE']:.3f} | R²={m_val_full['R2']:.4f}")
    print(f"  Teste:     RMSE={m_test_full['RMSE']:.3f} | MAE={m_test_full['MAE']:.3f} | R²={m_test_full['R2']:.4f}")
    print(f"  IC 95% Teste RMSE: {ci_test_full['RMSE_ci']}, MAE: {ci_test_full['MAE_ci']}")

    # ----------------------------------------------------------------------------------
    # 4. Treinamento do Modelo COMPAT (para validação cruzada com JAC iEV40)
    # ----------------------------------------------------------------------------------
    print("\n--- 4. Treinando Random Forest COMPAT ---")
    rf_compat = RandomForestRegressor(
        **best_params,
        random_state=SEED,
        n_jobs=-1
    )
    t0 = time.time()
    rf_compat.fit(X_train_compat, y_train)
    train_time_compat = time.time() - t0
    print(f"Treinamento RF COMPAT concluído em {train_time_compat:.1f}s")
    joblib.dump(rf_compat, MODELS / "rf_compat.joblib")

    p_test_compat = rf_compat.predict(X_test_compat)
    m_test_compat = metrics(y_test, p_test_compat)
    print(f"  Teste COMPAT: RMSE={m_test_compat['RMSE']:.3f} | MAE={m_test_compat['MAE']:.3f} | R²={m_test_compat['R2']:.4f}")

    # ----------------------------------------------------------------------------------
    # 5. Análise de Importância de Atributos (Feature Importance)
    # ----------------------------------------------------------------------------------
    print("\n--- 5. Extraindo e plotando Feature Importance ---")
    feat_imp = pd.Series(rf_full.feature_importances_, index=FEATURES_FULL).sort_values(ascending=False)
    feat_imp_df = pd.DataFrame({
        "feature": feat_imp.index,
        "importance": feat_imp.values,
        "label": [LABELS.get(f, f) for f in feat_imp.index]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="importance", y="label", data=feat_imp_df, palette="crest_r", ax=ax)
    ax.set_title("Importância de Atributos (MDI) — Random Forest Regressor", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importância Relativa (Mean Decrease in Impurity)")
    ax.set_ylabel("Atributo de Telemetria")
    plt.tight_layout()
    savefig(fig, "fig_rf_feature_importance")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 6. Gráficos de Resíduos e Real vs Predito
    # ----------------------------------------------------------------------------------
    print("--- 6. Gerando gráficos de resíduos e real vs predito ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Amostra de teste para visualização não sobrecarregar
    sample_sub = test_df.sample(n=min(8000, len(test_df)), random_state=SEED)
    sub_idx = sample_sub.index
    y_sub = y_test[sub_idx]
    p_sub = p_test_full[sub_idx]
    residuals = p_sub - y_sub

    # Scatter Real vs Predito
    ax1.scatter(y_sub, p_sub, alpha=0.3, s=10, color="#2E7D32")
    lims = [0, 100]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Predição Ideal (y = ŷ)")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title(f"Random Forest: Valor Real vs Predito (Teste)\nRMSE: {m_test_full['RMSE']:.2f}% | R²: {m_test_full['R2']:.4f}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("SOC Real [%]")
    ax1.set_ylabel("SOC Predito [%]")
    ax1.legend(frameon=True)

    # Distribuição dos Resíduos
    sns.histplot(residuals, bins=40, kde=True, color="#2E7D32", ax=ax2, alpha=0.6)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.axvline(np.mean(residuals), color="black", linestyle=":", linewidth=1.5, label=f"Viés Médio: {np.mean(residuals):.2f}%")
    ax2.set_title("Distribuição dos Resíduos no Conjunto de Teste", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Erro Residual (SOC Predito − SOC Real) [%]")
    ax2.set_ylabel("Frequência")
    ax2.legend(frameon=True)
    plt.tight_layout()
    savefig(fig, "fig_rf_residuals_and_scatter")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 7. Curvas Temporais de Predição vs Real em Viagens do Conjunto de Teste
    # ----------------------------------------------------------------------------------
    print("--- 7. Gerando perfis temporais em viagens de teste ---")
    test_trips = test_df.groupby("trip_id").filter(lambda x: len(x) > 150 and (x["soc"].max() - x["soc"].min() > 2))
    sample_trips = test_trips["trip_id"].unique()[:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, tid in zip(axes.flat, sample_trips):
        t_mask = (test_df["trip_id"] == tid).to_numpy()
        sub_t = test_df[t_mask].sort_values("ts_ms").reset_index(drop=True)
        time_min = (sub_t["ts_ms"] - sub_t["ts_ms"].iloc[0]) / 60000.0
        
        y_real = sub_t["soc"].to_numpy()
        X_t = scaler_full.transform(sub_t[FEATURES_FULL])
        y_pred = rf_full.predict(X_t)

        ax.plot(time_min, y_real, color="#263238", linewidth=2.5, label="SOC Real [%]")
        ax.plot(time_min, y_pred, color="#2E7D32", linestyle="--", linewidth=2.0, label="Random Forest")
        
        t_rmse = np.sqrt(np.mean((y_pred - y_real)**2))
        ax.set_title(f"Viagem ID: {tid} | RMSE local: {t_rmse:.2f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tempo decorrido [minutos]")
        ax.set_ylabel("SOC [%]")
        ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    savefig(fig, "fig_rf_temporal_predictions")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 8. Salvando Métricas Consolidadas
    # ----------------------------------------------------------------------------------
    rf_results = {
        "best_params": best_params,
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
        "feature_importances": feat_imp_df.to_dict(orient="records")
    }
    save_json(rf_results, "rf_metrics.json")
    print("\nProcessamento do Random Forest finalizado com sucesso!")

if __name__ == "__main__":
    run()
