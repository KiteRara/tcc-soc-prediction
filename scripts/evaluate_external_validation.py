"""
Avaliação da Generalização e Validação Externa no Electric Vehicle Telemetry Dataset (JAC iEV40 CAN)
conforme Seções 4.9 e Capítulo 5 do TCC.
Aplica os modelos RF COMPAT e XGBoost COMPAT diretamente nos dados CAN do veículo externo,
avalia o desempenho global e por sessão de condução (COND 1 a 9),
e gera gráficos diagnósticos da transferência de domínio e viabilidade embarcada.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common import (
    DATA_PROCESSED, FIGURES, MODELS, RESULTS, LABELS, TARGET,
    FEATURES_COMPAT, metrics, save_json, load_json, set_style, savefig
)

def run():
    set_style()
    print("--- 1. Carregando dataset externo e modelos COMPAT ---")
    ext_df = pd.read_parquet(DATA_PROCESSED / "test_external.parquet")
    print(f"Dataset externo carregado: {ext_df.shape}")

    scaler_compat = joblib.load(MODELS / "scaler_compat.joblib")
    rf_compat = joblib.load(MODELS / "rf_compat.joblib")
    xgb_compat = joblib.load(MODELS / "xgb_compat.joblib")

    # Features compatíveis normalizadas com o scaler do treino
    X_ext = scaler_compat.transform(ext_df[FEATURES_COMPAT])
    y_ext = ext_df[TARGET].to_numpy()

    # ----------------------------------------------------------------------------------
    # 2. Inferência Direta dos Modelos
    # ----------------------------------------------------------------------------------
    print("\n--- 2. Executando inferência nos dados CAN do JAC iEV40 ---")
    p_rf = rf_compat.predict(X_ext)
    p_xgb = xgb_compat.predict(X_ext)

    ext_df["pred_rf"] = p_rf
    ext_df["pred_xgb"] = p_xgb
    ext_df["err_rf"] = p_rf - y_ext
    ext_df["err_xgb"] = p_xgb - y_ext

    # Métricas Globais (todas as 24.277 amostras)
    m_rf_all = metrics(y_ext, p_rf)
    m_xgb_all = metrics(y_ext, p_xgb)

    # Subconjunto com Tensão Válida de Alta Tensão (~350 V, Sessões 7 e 8)
    valid_mask = ext_df["voltage_valid"].to_numpy()
    m_rf_valid = metrics(y_ext[valid_mask], p_rf[valid_mask])
    m_xgb_valid = metrics(y_ext[valid_mask], p_xgb[valid_mask])

    print("Resultados Globais (todas as sessões):")
    print(f"  RF:  RMSE={m_rf_all['RMSE']:.2f}% | MAE={m_rf_all['MAE']:.2f}% | R²={m_rf_all['R2']:.4f}")
    print(f"  XGB: RMSE={m_xgb_all['RMSE']:.2f}% | MAE={m_xgb_all['MAE']:.2f}% | R²={m_xgb_all['R2']:.4f}")

    print("\nResultados nas Sessões de Alta Tensão Válida (~350V, COND 7 e 8, N=6.162):")
    print(f"  RF:  RMSE={m_rf_valid['RMSE']:.2f}% | MAE={m_rf_valid['MAE']:.2f}% | R²={m_rf_valid['R2']:.4f}")
    print(f"  XGB: RMSE={m_xgb_valid['RMSE']:.2f}% | MAE={m_xgb_valid['MAE']:.2f}% | R²={m_xgb_valid['R2']:.4f}")

    # ----------------------------------------------------------------------------------
    # 3. Desempenho por Sessão de Condução (COND 1 a 9)
    # ----------------------------------------------------------------------------------
    print("\n--- 3. Avaliando desempenho por sessão (COND 1 a 9) ---")
    cond_stats = []
    for cond_val, group in ext_df.groupby("cond"):
        y_c = group[TARGET].to_numpy()
        p_rf_c = group["pred_rf"].to_numpy()
        p_xgb_c = group["pred_xgb"].to_numpy()
        v_mean = float(group["voltage"].mean())
        cond_stats.append({
            "COND": int(cond_val),
            "N": len(group),
            "Tensão_Média_V": round(v_mean, 1),
            "SOC_Real_Médio": round(float(y_c.mean()), 1),
            "RF_RMSE": round(float(np.sqrt(np.mean((p_rf_c - y_c)**2))), 2),
            "RF_MAE": round(float(np.mean(np.abs(p_rf_c - y_c))), 2),
            "XGB_RMSE": round(float(np.sqrt(np.mean((p_xgb_c - y_c)**2))), 2),
            "XGB_MAE": round(float(np.mean(np.abs(p_xgb_c - y_c))), 2),
            "Autonomia_Média_km": round(float(group["autonomy"].mean()), 1)
        })
    cond_df = pd.DataFrame(cond_stats)
    print(cond_df.to_string(index=False))

    # ----------------------------------------------------------------------------------
    # 4. Gráficos de Validação Externa
    # ----------------------------------------------------------------------------------
    print("\n--- 4. Gerando gráficos de validação externa ---")
    
    # 4.1 Curvas Temporais nas Sessões Válidas (COND 7 e 8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for ax, cond_id, title in [(ax1, 7, "Sessão CAN 7 (JAC iEV40)"), (ax2, 8, "Sessão CAN 8 (JAC iEV40)")]:
        sub = ext_df[ext_df["cond"] == cond_id].reset_index(drop=True)
        t_seq = sub["seq"].to_numpy()
        
        ax.plot(t_seq, sub[TARGET], color="#263238", linewidth=2.8, label="SOC Real (CH) [%]")
        ax.plot(t_seq, sub["pred_rf"], color="#2E7D32", linestyle="--", linewidth=2.0, label="Random Forest")
        ax.plot(t_seq, sub["pred_xgb"], color="#C62828", linestyle="-.", linewidth=2.0, label="XGBoost")
        
        rmse_rf = np.sqrt(np.mean((sub["pred_rf"] - sub[TARGET])**2))
        rmse_xgb = np.sqrt(np.mean((sub["pred_xgb"] - sub[TARGET])**2))
        
        ax.set_title(f"{title}\nRF RMSE: {rmse_rf:.2f}% | XGB RMSE: {rmse_xgb:.2f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Amostra Temporal Contínua (CAN)")
        ax.set_ylabel("SOC [%]")
        ax.legend(frameon=True, loc="upper right")
        ax.set_ylim(10, 100)
    plt.tight_layout()
    savefig(fig, "fig_external_temporal_sessions_7_8")
    plt.close(fig)

    # 4.2 Scatter Real vs Predito nas Sessões Válidas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sub_val = ext_df[valid_mask]

    ax1.scatter(sub_val[TARGET], sub_val["pred_rf"], alpha=0.3, s=12, color="#2E7D32")
    lims = [15, 90]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Predição Ideal")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title(f"Random Forest no JAC iEV40 (COND 7 e 8)\nRMSE: {m_rf_valid['RMSE']:.2f}% | R²: {m_rf_valid['R2']:.4f}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("SOC Real do Painel CAN (CH) [%]")
    ax1.set_ylabel("SOC Predito pelo Modelo [%]")
    ax1.legend(frameon=True)

    ax2.scatter(sub_val[TARGET], sub_val["pred_xgb"], alpha=0.3, s=12, color="#C62828")
    ax2.plot(lims, lims, "k--", linewidth=1.5, label="Predição Ideal")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_title(f"XGBoost no JAC iEV40 (COND 7 e 8)\nRMSE: {m_xgb_valid['RMSE']:.2f}% | R²: {m_xgb_valid['R2']:.4f}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("SOC Real do Painel CAN (CH) [%]")
    ax2.set_ylabel("SOC Predito pelo Modelo [%]")
    ax2.legend(frameon=True)
    plt.tight_layout()
    savefig(fig, "fig_external_scatter_valid_sessions")
    plt.close(fig)

    # 4.3 Análise Comparativa do Erro por Sessão (COND 1 a 9)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_pos = np.arange(len(cond_df))
    width = 0.35

    ax.bar(x_pos - width/2, cond_df["RF_RMSE"], width, label="Random Forest (RMSE)", color="#2E7D32", alpha=0.85, edgecolor="black")
    ax.bar(x_pos + width/2, cond_df["XGB_RMSE"], width, label="XGBoost (RMSE)", color="#C62828", alpha=0.85, edgecolor="black")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Sessão {c}\n({v}V)" for c, v in zip(cond_df["COND"], cond_df["Tensão_Média_V"])], fontsize=10)
    ax.set_ylabel("RMSE de Predição do SOC [%]", fontsize=11)
    ax.set_title("RMSE dos Modelos em cada Sessão de Coleta CAN (JAC iEV40)", fontsize=12, fontweight="bold")
    ax.axvspan(5.5, 7.5, color="green", alpha=0.15, label="Sessões com Pack HV Válido (~350V)")
    ax.legend(frameon=True)
    plt.tight_layout()
    savefig(fig, "fig_external_rmse_by_session")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 5. Salvando Métricas de Validação Externa
    # ----------------------------------------------------------------------------------
    ext_results = {
        "global_metrics": {
            "RandomForest": m_rf_all,
            "XGBoost": m_xgb_all
        },
        "valid_voltage_sessions_metrics": {
            "RandomForest": m_rf_valid,
            "XGBoost": m_xgb_valid
        },
        "per_session_metrics": cond_df.to_dict(orient="records")
    }
    save_json(ext_results, "external_validation_metrics.json")
    print("\nValidação Externa concluída com sucesso!")

if __name__ == "__main__":
    run()
