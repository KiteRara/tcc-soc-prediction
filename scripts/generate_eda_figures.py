"""
Gera todas as figuras da Análise Exploratória dos Dados (EDA) conforme Seção 4.7 do TCC.
Salva em figures/ tanto em PNG (300 DPI) quanto em PDF vetorial.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common import (
    DATA_PROCESSED, FIGURES, LABELS, set_style, savefig
)

def run():
    set_style()
    print("Carregando train.parquet para EDA...")
    df = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    print(f"Dados carregados: {df.shape}")

    # ----------------------------------------------------------------------------------
    # 1. Distribuições Univariadas das Variáveis Principais
    # ----------------------------------------------------------------------------------
    print("1. Gerando distribuições univariadas...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    vars_plot = [
        ("soc", "Estado de Carga (SOC) [%]", "#1976D2"),
        ("voltage", "Tensão HV [V]", "#388E3C"),
        ("current", "Corrente HV [A]", "#F57C00"),
        ("speed", "Velocidade [km/h]", "#7B1FA2"),
        ("oat", "Temperatura Externa (OAT) [°C]", "#00796B"),
        ("ac_power_w", "Potência do A/C [W]", "#D32F2F")
    ]
    for ax, (col, label, color) in zip(axes.flat, vars_plot):
        data = df[col].dropna()
        sns.histplot(data, bins=50, kde=True, ax=ax, color=color, edgecolor="none", alpha=0.6)
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5, label=f"Média: {mean_val:.1f}")
        ax.axvline(median_val, color="red", linestyle=":", linewidth=1.5, label=f"Mediana: {median_val:.1f}")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frequência")
        ax.legend(frameon=True)
    plt.tight_layout()
    savefig(fig, "fig_eda_univariate_distributions")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 2. Matriz de Correlação de Pearson com o SOC
    # ----------------------------------------------------------------------------------
    print("2. Gerando matriz de correlação...")
    corr_cols = [
        "soc", "voltage", "current", "speed", "power_kw",
        "current_ma5", "voltage_ma5", "speed_ma5",
        "oat", "ac_power_w", "heater_w", "elevation", "gradient", "energy_cons"
    ]
    corr_matrix = df[corr_cols].corr(method="pearson")
    corr_labels = [LABELS.get(c, c) for c in corr_cols]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, center=0, cbar_kws={"label": "Coeficiente de Pearson (r)"},
        xticklabels=corr_labels, yticklabels=corr_labels, ax=ax, square=True,
        annot_kws={"size": 9}
    )
    ax.set_title("Matriz de Correlação de Pearson — Variáveis de Telemetria e SOC", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    savefig(fig, "fig_eda_correlation_matrix")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 3. Perfis Temporais do SOC em Viagens Reais
    # ----------------------------------------------------------------------------------
    print("3. Gerando perfis temporais de viagens...")
    # Selecionar 4 viagens com perfis distintos e número razoável de amostras
    top_trips = df.groupby("trip_id").filter(lambda x: len(x) > 200 and (x["soc"].max() - x["soc"].min() > 3))
    sample_trip_ids = top_trips["trip_id"].unique()[:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    for ax, tid in zip(axes.flat, sample_trip_ids):
        tdf = df[df["trip_id"] == tid].sort_values("ts_ms").copy()
        time_min = (tdf["ts_ms"] - tdf["ts_ms"].iloc[0]) / 60000.0  # minutos

        ax2 = ax.twinx()
        l1 = ax.plot(time_min, tdf["soc"], color="#1976D2", linewidth=2.2, label="SOC [%]")
        l2 = ax2.plot(time_min, tdf["speed"], color="#9E9E9E", alpha=0.5, linewidth=1.2, label="Velocidade [km/h]")
        
        ax.set_title(f"Viagem ID: {tid} (Veículo {int(tdf['veh'].iloc[0])})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tempo decorrido [minutos]")
        ax.set_ylabel("SOC [%]", color="#1976D2")
        ax2.set_ylabel("Velocidade [km/h]", color="#616161")
        ax2.grid(False)

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", frameon=True)
    plt.tight_layout()
    savefig(fig, "fig_eda_temporal_trips_profile")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 4. Relação Tensão vs SOC e Impacto da Corrente
    # ----------------------------------------------------------------------------------
    print("4. Gerando scatter Tensão vs SOC colorido pela corrente...")
    sample_sub = df.sample(n=min(15000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        sample_sub["voltage"], sample_sub["soc"],
        c=sample_sub["current"], cmap="Spectral", s=12, alpha=0.6
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Corrente HV [A] (negativo = descarga, positivo = regeneração)")
    ax.set_title("Relação não-linear Tensão HV vs SOC em Condições Reais de Condução", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tensão da Bateria HV [V]")
    ax.set_ylabel("Estado de Carga (SOC) [%]")
    plt.tight_layout()
    savefig(fig, "fig_eda_voltage_soc_scatter")
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # 5. Impacto de Sistemas Auxiliares (A/C e Aquecedor) e Temperatura no Consumo
    # ----------------------------------------------------------------------------------
    print("5. Gerando gráfico de impacto de sistemas auxiliares...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # A/C ligado vs desligado
    df_ac = df.copy()
    df_ac["ac_status"] = np.where(df_ac["ac_on"] == 1, "A/C Ligado", "A/C Desligado")
    sns.boxplot(x="ac_status", y="power_kw", data=df_ac.sample(20000, random_state=42), ax=ax1, palette=["#4CAF50", "#FF9800"])
    ax1.set_title("Potência Total Demandada da Bateria com A/C", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Status do Ar-Condicionado")
    ax1.set_ylabel("Potência Elétrica da Bateria [kW]")

    # Impacto da temperatura externa no aquecedor
    sns.scatterplot(
        x="oat", y="heater_w", data=df.sample(15000, random_state=42),
        alpha=0.4, color="#E53935", ax=ax2
    )
    ax2.set_title("Potência do Aquecedor em Função da Temperatura Externa (OAT)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Temperatura Ambiente Exterior (OAT) [°C]")
    ax2.set_ylabel("Potência do Aquecedor [W]")
    
    plt.tight_layout()
    savefig(fig, "fig_eda_auxiliary_systems_impact")
    plt.close(fig)

    print("Todas as figuras de EDA foram geradas com sucesso!")

if __name__ == "__main__":
    run()
