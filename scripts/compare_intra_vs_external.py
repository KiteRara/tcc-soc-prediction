"""
Experimento comparativo: Treinamento + Validação/Teste APENAS no dataset de treinamento (VED)
versus Validação Externa em outro dataset (JAC iEV40 CAN).

Avalia três níveis progressivos de generalização reutilizando exatamente o pipeline
de src/common.py e os hiperparâmetros já otimizados no TCC:
  1. Cenário 1 (Intra-dataset Temporal — VED -> VED, viagens futuras dos mesmos veículos)
  2. Cenário 2 (Intra-dataset por Veículo — VED -> VED, veículos inéditos / 0% overlap de VehId)
  3. Cenário 3 (Cross-dataset / Validação Externa — VED -> JAC iEV40 CAN)

Salva métricas em results/intra_vs_external_comparison.json e gera figuras em figures/.
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
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from src.common import (
    DATA_PROCESSED,
    FEATURES_COMPAT,
    FEATURES_FULL,
    FIGURES,
    MODELS,
    PALETTE,
    RESULTS,
    SEED,
    TARGET,
    bootstrap_ci,
    load_json,
    metrics,
    save_json,
    savefig,
    set_style,
)


def stratified_vehicle_split(df: pd.DataFrame) -> pd.Series:
    """
    Divide os 28 veículos do VED em train (~70%), val (~15%) e test (~15%)
    com 0% de sobreposição de veículos, preservando as três arquiteturas de
    tensão nominal do pack (~200 V, ~300 V e ~350-385 V) em todas as partições.
    """
    veh_stats = (
        df.groupby("veh")
        .agg(n=("soc", "size"), vmed=("voltage", "median"))
        .sort_values("n", ascending=False)
    )

    # Três faixas físicas de arquitetura de bateria presentes no VED
    def voltage_group(v: float) -> str:
        if v < 250:
            return "~200 V"
        if v < 330:
            return "~300 V"
        return "~360 V"

    veh_stats["vgroup"] = veh_stats["vmed"].map(voltage_group)

    # Alocação determinística estratificada por faixa de tensão (~70/15/15 em amostras)
    val_vehs = {561.0, 453.0, 457.0, 536.0, 567.0}
    test_vehs = {398.0, 11.0, 537.0, 492.0, 542.0}
    mapping = {}
    for veh_id in veh_stats.index:
        if veh_id in test_vehs:
            mapping[veh_id] = "test"
        elif veh_id in val_vehs:
            mapping[veh_id] = "val"
        else:
            mapping[veh_id] = "train"

    return df["veh"].map(mapping)


def run():
    set_style()
    print("=== 1. Carregando dados processados e modelos existentes (Cenário 1: Temporal VED) ===")
    train_temp = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    val_temp = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    test_temp = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    ext_df = pd.read_parquet(DATA_PROCESSED / "test_external.parquet")

    rf_meta = load_json("rf_metrics.json")
    xgb_meta = load_json("xgb_metrics.json")
    ext_meta = load_json("external_validation_metrics.json")

    scaler_full = joblib.load(MODELS / "scaler_full.joblib")
    scaler_compat = joblib.load(MODELS / "scaler_compat.joblib")
    rf_full = joblib.load(MODELS / "rf_full.joblib")
    rf_compat = joblib.load(MODELS / "rf_compat.joblib")
    xgb_full = joblib.load(MODELS / "xgb_full.joblib")
    xgb_compat = joblib.load(MODELS / "xgb_compat.joblib")

    # Completar métricas COMPAT no Cenário 1 (train, val, test + IC 95% bootstrap)
    c1 = {"full": {}, "compat": {}}
    for feat_name, cols, sc, m_rf, m_xgb in [
        ("full", FEATURES_FULL, scaler_full, rf_full, xgb_full),
        ("compat", FEATURES_COMPAT, scaler_compat, rf_compat, xgb_compat),
    ]:
        X_tr, y_tr = sc.transform(train_temp[cols]), train_temp[TARGET].to_numpy()
        X_va, y_va = sc.transform(val_temp[cols]), val_temp[TARGET].to_numpy()
        X_te, y_te = sc.transform(test_temp[cols]), test_temp[TARGET].to_numpy()

        p_rf_va, p_rf_te = m_rf.predict(X_va), m_rf.predict(X_te)
        p_xgb_va, p_xgb_te = m_xgb.predict(X_va), m_xgb.predict(X_te)

        c1[feat_name] = {
            "RF": {
                "val": metrics(y_va, p_rf_va),
                "test": metrics(y_te, p_rf_te),
                "test_ci": bootstrap_ci(y_te, p_rf_te, test_temp["trip_id"]),
            },
            "XGB": {
                "val": metrics(y_va, p_xgb_va),
                "test": metrics(y_te, p_xgb_te),
                "test_ci": bootstrap_ci(y_te, p_xgb_te, test_temp["trip_id"]),
            },
        }
        if feat_name == "compat":
            test_temp = test_temp.copy()
            test_temp["pred_rf_c1"] = p_rf_te
            test_temp["pred_xgb_c1"] = p_xgb_te

    print("Cenário 1 (VED Temporal — COMPAT):")
    print(f"  RF  Test: RMSE={c1['compat']['RF']['test']['RMSE']:.2f}% | MAE={c1['compat']['RF']['test']['MAE']:.2f}% | R²={c1['compat']['RF']['test']['R2']:.4f}")
    print(f"  XGB Test: RMSE={c1['compat']['XGB']['test']['RMSE']:.2f}% | MAE={c1['compat']['XGB']['test']['MAE']:.2f}% | R²={c1['compat']['XGB']['test']['R2']:.4f}")

    # ==================================================================================
    # 2. Cenário 2: Treinamento + Validação/Teste no VED com Veículos Inéditos (GroupSplit)
    # ==================================================================================
    print("\n=== 2. Cenário 2: Divisão por Veículos Inéditos dentro do VED (0% overlap) ===")
    ved_all = pd.concat([train_temp, val_temp, test_temp.drop(columns=["pred_rf_c1", "pred_xgb_c1"])], ignore_index=True)
    ved_all["veh_split"] = stratified_vehicle_split(ved_all)

    tr_v = ved_all[ved_all["veh_split"] == "train"].reset_index(drop=True)
    va_v = ved_all[ved_all["veh_split"] == "val"].reset_index(drop=True)
    te_v = ved_all[ved_all["veh_split"] == "test"].reset_index(drop=True)

    # Verificação estrita de vazamento zero entre veículos
    assert set(tr_v["veh"]).isdisjoint(set(va_v["veh"]))
    assert set(tr_v["veh"]).isdisjoint(set(te_v["veh"]))
    assert set(va_v["veh"]).isdisjoint(set(te_v["veh"]))

    split_summary = {
        "train": {"n": len(tr_v), "frac": len(tr_v) / len(ved_all), "n_vehs": int(tr_v["veh"].nunique()), "vehs": sorted(int(v) for v in tr_v["veh"].unique())},
        "val": {"n": len(va_v), "frac": len(va_v) / len(ved_all), "n_vehs": int(va_v["veh"].nunique()), "vehs": sorted(int(v) for v in va_v["veh"].unique())},
        "test": {"n": len(te_v), "frac": len(te_v) / len(ved_all), "n_vehs": int(te_v["veh"].nunique()), "vehs": sorted(int(v) for v in te_v["veh"].unique())},
    }
    for k, v in split_summary.items():
        print(f"  {k:>5}: {v['n']:>7,} amostras ({v['frac']*100:4.1f}%) | {v['n_vehs']:>2} veículos: {v['vehs']}")

    y_tr_v = tr_v[TARGET].to_numpy()
    y_va_v = va_v[TARGET].to_numpy()
    y_te_v = te_v[TARGET].to_numpy()

    rf_params = rf_meta["best_params"]
    xgb_params = {
        **xgb_meta["best_params"],
        "n_estimators": 500,
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "early_stopping_rounds": 25,
    }

    c2 = {"full": {}, "compat": {}}
    for feat_name, cols in [("full", FEATURES_FULL), ("compat", FEATURES_COMPAT)]:
        print(f"\n  Treinando Cenário 2 ({feat_name.upper()}, {len(cols)} features)...")
        sc = MinMaxScaler(feature_range=(0, 1)).fit(tr_v[cols])
        X_tr = sc.transform(tr_v[cols])
        X_va = sc.transform(va_v[cols])
        X_te = sc.transform(te_v[cols])

        # Random Forest
        t0 = time.time()
        rf_v = RandomForestRegressor(**rf_params, random_state=SEED, n_jobs=-1)
        rf_v.fit(X_tr, y_tr_v)
        rf_t = time.time() - t0
        joblib.dump(rf_v, MODELS / f"rf_vehsplit_{feat_name}.joblib")

        p_rf_tr = rf_v.predict(X_tr)
        p_rf_va = rf_v.predict(X_va)
        p_rf_te = rf_v.predict(X_te)

        # XGBoost com early stopping na partição de validação por veículos inéditos
        t0 = time.time()
        xgb_v = XGBRegressor(**xgb_params)
        xgb_v.fit(X_tr, y_tr_v, eval_set=[(X_va, y_va_v)], verbose=False)
        xgb_t = time.time() - t0
        joblib.dump(xgb_v, MODELS / f"xgb_vehsplit_{feat_name}.joblib")

        p_xgb_tr = xgb_v.predict(X_tr)
        p_xgb_va = xgb_v.predict(X_va)
        p_xgb_te = xgb_v.predict(X_te)

        c2[feat_name] = {
            "RF": {
                "train": metrics(y_tr_v, p_rf_tr),
                "val": metrics(y_va_v, p_rf_va),
                "test": metrics(y_te_v, p_rf_te),
                "test_ci": bootstrap_ci(y_te_v, p_rf_te, te_v["trip_id"]),
                "train_time_s": rf_t,
            },
            "XGB": {
                "train": metrics(y_tr_v, p_xgb_tr),
                "val": metrics(y_va_v, p_xgb_va),
                "test": metrics(y_te_v, p_xgb_te),
                "test_ci": bootstrap_ci(y_te_v, p_xgb_te, te_v["trip_id"]),
                "best_iteration": int(xgb_v.best_iteration),
                "train_time_s": xgb_t,
            },
        }
        print(f"    RF  ({rf_t:4.1f}s) -> Val RMSE={c2[feat_name]['RF']['val']['RMSE']:.2f}% | Test RMSE={c2[feat_name]['RF']['test']['RMSE']:.2f}% | MAE={c2[feat_name]['RF']['test']['MAE']:.2f}% | R²={c2[feat_name]['RF']['test']['R2']:.4f}")
        print(f"    XGB ({xgb_t:4.1f}s, it={xgb_v.best_iteration}) -> Val RMSE={c2[feat_name]['XGB']['val']['RMSE']:.2f}% | Test RMSE={c2[feat_name]['XGB']['test']['RMSE']:.2f}% | MAE={c2[feat_name]['XGB']['test']['MAE']:.2f}% | R²={c2[feat_name]['XGB']['test']['R2']:.4f}")

        if feat_name == "compat":
            te_v["pred_rf_c2"] = p_rf_te
            te_v["pred_xgb_c2"] = p_xgb_te

    # ==================================================================================
    # 3. Métricas por veículo (Cenário 1 vs Cenário 2 — COMPAT)
    # ==================================================================================
    per_veh_c1 = []
    for veh_id, g in test_temp.groupby("veh"):
        per_veh_c1.append({
            "veh": int(veh_id), "n": len(g), "vmed": round(float(g["voltage"].median()), 1),
            "RF_RMSE": metrics(g[TARGET], g["pred_rf_c1"])["RMSE"],
            "XGB_RMSE": metrics(g[TARGET], g["pred_xgb_c1"])["RMSE"],
            "XGB_MAE": metrics(g[TARGET], g["pred_xgb_c1"])["MAE"],
            "XGB_R2": metrics(g[TARGET], g["pred_xgb_c1"])["R2"],
        })
    per_veh_c2 = []
    for veh_id, g in te_v.groupby("veh"):
        per_veh_c2.append({
            "veh": int(veh_id), "n": len(g), "vmed": round(float(g["voltage"].median()), 1),
            "RF_RMSE": metrics(g[TARGET], g["pred_rf_c2"])["RMSE"],
            "XGB_RMSE": metrics(g[TARGET], g["pred_xgb_c2"])["RMSE"],
            "XGB_MAE": metrics(g[TARGET], g["pred_xgb_c2"])["MAE"],
            "XGB_R2": metrics(g[TARGET], g["pred_xgb_c2"])["R2"],
        })
    df_pv_c1 = pd.DataFrame(per_veh_c1).sort_values("vmed")
    df_pv_c2 = pd.DataFrame(per_veh_c2).sort_values("vmed")

    print("\nDesempenho por veículo inédito no Cenário 2 (COMPAT):")
    print(df_pv_c2.round(2).to_string(index=False))

    # ==================================================================================
    # 4. Cenário 3: Validação Externa (JAC iEV40) com IC 95% por blocos temporais
    # ==================================================================================
    X_ext = scaler_compat.transform(ext_df[FEATURES_COMPAT])
    y_ext = ext_df[TARGET].to_numpy()
    p_rf_ext = rf_compat.predict(X_ext)
    p_xgb_ext = xgb_compat.predict(X_ext)

    # Criar blocos de 150 amostras consecutivas dentro de cada sessão para bootstrap em blocos
    ext_blocks = ext_df["cond"].astype(int) * 1000 + (ext_df["seq"] // 150)
    valid_mask = ext_df["voltage_valid"].to_numpy()
    c7_mask = (ext_df["cond"] == 7).to_numpy()

    c3 = {
        "session7": {
            "RF": metrics(y_ext[c7_mask], p_rf_ext[c7_mask]),
            "RF_ci": bootstrap_ci(y_ext[c7_mask], p_rf_ext[c7_mask], ext_blocks[c7_mask]),
            "XGB": metrics(y_ext[c7_mask], p_xgb_ext[c7_mask]),
            "XGB_ci": bootstrap_ci(y_ext[c7_mask], p_xgb_ext[c7_mask], ext_blocks[c7_mask]),
        },
        "valid_7_8": {
            "RF": ext_meta["valid_voltage_sessions_metrics"]["RandomForest"],
            "RF_ci": bootstrap_ci(y_ext[valid_mask], p_rf_ext[valid_mask], ext_blocks[valid_mask]),
            "XGB": ext_meta["valid_voltage_sessions_metrics"]["XGBoost"],
            "XGB_ci": bootstrap_ci(y_ext[valid_mask], p_xgb_ext[valid_mask], ext_blocks[valid_mask]),
        },
        "global_all": {
            "RF": ext_meta["global_metrics"]["RandomForest"],
            "RF_ci": bootstrap_ci(y_ext, p_rf_ext, ext_blocks),
            "XGB": ext_meta["global_metrics"]["XGBoost"],
            "XGB_ci": bootstrap_ci(y_ext, p_xgb_ext, ext_blocks),
        },
    }

    # ==================================================================================
    # 5. Figuras Comparativas (PNG 300 DPI + PDF vetorial)
    # ==================================================================================
    print("\n=== 3. Gerando figuras comparativas (Intra-dataset vs Validação Externa) ===")

    # Figura A: Comparação direta RMSE e MAE entre os cenários (mesmas 10 features COMPAT)
    scenarios = [
        ("Cenário 1\nVED Temporal\n(mesmos veículos)", c1["compat"]["RF"]["test"], c1["compat"]["RF"]["test_ci"], c1["compat"]["XGB"]["test"], c1["compat"]["XGB"]["test_ci"]),
        ("Cenário 2\nVED Veículos\nInéditos", c2["compat"]["RF"]["test"], c2["compat"]["RF"]["test_ci"], c2["compat"]["XGB"]["test"], c2["compat"]["XGB"]["test_ci"]),
        ("Cenário 3a\nJAC iEV40\n(Sessão 7, ~362V)", c3["session7"]["RF"], c3["session7"]["RF_ci"], c3["session7"]["XGB"], c3["session7"]["XGB_ci"]),
        ("Cenário 3b\nJAC iEV40\n(Sessões 7–8, HV)", c3["valid_7_8"]["RF"], c3["valid_7_8"]["RF_ci"], c3["valid_7_8"]["XGB"], c3["valid_7_8"]["XGB_ci"]),
        ("Cenário 3c\nJAC iEV40\n(Global, c/ 12V)", c3["global_all"]["RF"], c3["global_all"]["RF_ci"], c3["global_all"]["XGB"], c3["global_all"]["XGB_ci"]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4))
    x = np.arange(len(scenarios))
    w = 0.36

    for ax, metric_key, ci_key, ylabel, title in [
        (ax1, "RMSE", "RMSE_ci", "RMSE [% de SOC]", "RMSE por Cenário de Validação (Atributos COMPAT)"),
        (ax2, "MAE", "MAE_ci", "MAE [% de SOC]", "MAE por Cenário de Validação (Atributos COMPAT)"),
    ]:
        rf_vals = [s[1][metric_key] for s in scenarios]
        rf_err = [
            [max(0.0, s[1][metric_key] - s[2][ci_key][0]) for s in scenarios],
            [max(0.0, s[2][ci_key][1] - s[1][metric_key]) for s in scenarios],
        ]
        xgb_vals = [s[3][metric_key] for s in scenarios]
        xgb_err = [
            [max(0.0, s[3][metric_key] - s[4][ci_key][0]) for s in scenarios],
            [max(0.0, s[4][ci_key][1] - s[3][metric_key]) for s in scenarios],
        ]

        b1 = ax.bar(x - w / 2, rf_vals, w, yerr=rf_err, capsize=4, label="Random Forest", color=PALETTE["rf"], alpha=0.88, edgecolor="black")
        b2 = ax.bar(x + w / 2, xgb_vals, w, yerr=xgb_err, capsize=4, label="XGBoost", color=PALETTE["xgb"], alpha=0.88, edgecolor="black")

        top_hi = max(v + e for v, e in zip(rf_vals + xgb_vals, rf_err[1] + xgb_err[1]))
        for bars, err_hi in [(b1, rf_err[1]), (b2, xgb_err[1])]:
            for bar, eh in zip(bars, err_hi):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + eh + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.axvline(1.5, color="#455A64", linestyle="--", linewidth=1.3)
        ax.axvspan(-0.5, 1.5, color="#E8F5E9", alpha=0.35, zorder=0)
        ax.axvspan(1.5, 4.5, color="#FFEBEE", alpha=0.28, zorder=0)
        ax.text(0.5, top_hi * 1.16, "Intra-dataset (VED → VED)", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1B5E20")
        ax.text(3.0, top_hi * 1.16, "Cross-dataset (VED → JAC iEV40)", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#B71C1C")
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in scenarios], fontsize=8.8)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_ylim(0, top_hi * 1.25)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, fontsize=10.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_intra_vs_external_comparison")
    plt.close(fig)

    # Figura B: Desempenho por veículo no VED (Temporal vs Veículos Inéditos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.8, 1.0]})

    # Painel esquerdo: RMSE por veículo no Cenário 1 (Temporal)
    x1 = np.arange(len(df_pv_c1))
    ax1.bar(x1 - 0.18, df_pv_c1["RF_RMSE"], 0.36, label="Random Forest", color=PALETTE["rf"], alpha=0.85, edgecolor="black")
    ax1.bar(x1 + 0.18, df_pv_c1["XGB_RMSE"], 0.36, label="XGBoost", color=PALETTE["xgb"], alpha=0.85, edgecolor="black")
    ax1.axhline(c1["compat"]["XGB"]["test"]["RMSE"], color=PALETTE["xgb"], linestyle=":", linewidth=1.5, label=f"Média global XGB ({c1['compat']['XGB']['test']['RMSE']:.1f}%)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([f"V{int(v)}\n({vm:.0f}V)" for v, vm in zip(df_pv_c1["veh"], df_pv_c1["vmed"])], fontsize=7.5, rotation=45, ha="right")
    ax1.set_ylabel("RMSE [% de SOC]")
    ax1.set_title("Cenário 1 (VED Temporal): RMSE por Veículo", fontsize=11, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.5)

    # Painel direito: RMSE por veículo inédito no Cenário 2 (GroupSplit)
    x2 = np.arange(len(df_pv_c2))
    b_rf = ax2.bar(x2 - 0.18, df_pv_c2["RF_RMSE"], 0.36, label="Random Forest", color=PALETTE["rf"], alpha=0.85, edgecolor="black")
    b_xgb = ax2.bar(x2 + 0.18, df_pv_c2["XGB_RMSE"], 0.36, label="XGBoost", color=PALETTE["xgb"], alpha=0.85, edgecolor="black")
    for bars in (b_rf, b_xgb):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.axhline(c2["compat"]["XGB"]["test"]["RMSE"], color=PALETTE["xgb"], linestyle=":", linewidth=1.5, label=f"Média global XGB ({c2['compat']['XGB']['test']['RMSE']:.1f}%)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"V{int(v)}\n({vm:.0f}V)" for v, vm in zip(df_pv_c2["veh"], df_pv_c2["vmed"])], fontsize=8.8)
    ax2.set_ylabel("RMSE [% de SOC]")
    ax2.set_title("Cenário 2 (VED Veículos Inéditos): RMSE por Veículo", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(df_pv_c2["RF_RMSE"].max(), df_pv_c2["XGB_RMSE"].max()) * 1.18)
    ax2.legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    savefig(fig, "fig_ved_per_vehicle_comparison")
    plt.close(fig)

    # Figura C: Curvas temporais em 4 viagens de veículos inéditos (Cenário 2 — COMPAT)
    cand_trips = te_v.groupby("trip_id").filter(lambda g: len(g) > 150 and (g["soc"].max() - g["soc"].min() > 3))
    # Selecionar até 1 viagem de cada veículo inédito (4 veículos diferentes)
    chosen_trips = []
    for vid in [537.0, 492.0, 11.0, 398.0]:
        sub_v = cand_trips[cand_trips["veh"] == vid]
        if len(sub_v) > 0:
            tid = sub_v.groupby("trip_id").size().idxmax()
            chosen_trips.append((int(vid), tid))
    if len(chosen_trips) < 4:
        for tid in cand_trips["trip_id"].unique():
            if all(t != tid for _, t in chosen_trips):
                vid = int(cand_trips[cand_trips["trip_id"] == tid]["veh"].iloc[0])
                chosen_trips.append((vid, tid))
            if len(chosen_trips) == 4:
                break

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    for ax, (vid, tid) in zip(axes.flat, chosen_trips):
        sub_t = te_v[te_v["trip_id"] == tid].sort_values("ts_ms").reset_index(drop=True)
        time_min = (sub_t["ts_ms"] - sub_t["ts_ms"].iloc[0]) / 60000.0
        y_real = sub_t[TARGET].to_numpy()
        y_rf = sub_t["pred_rf_c2"].to_numpy()
        y_xgb = sub_t["pred_xgb_c2"].to_numpy()
        vmed = sub_t["voltage"].median()

        rmse_rf = np.sqrt(np.mean((y_rf - y_real) ** 2))
        rmse_xgb = np.sqrt(np.mean((y_xgb - y_real) ** 2))

        ax.plot(time_min, y_real, color=PALETTE["real"], linewidth=2.6, label="SOC Real [%]")
        ax.plot(time_min, y_rf, color=PALETTE["rf"], linestyle="--", linewidth=1.8, label=f"Random Forest (RMSE: {rmse_rf:.2f}%)")
        ax.plot(time_min, y_xgb, color=PALETTE["xgb"], linestyle="-.", linewidth=1.8, label=f"XGBoost (RMSE: {rmse_xgb:.2f}%)")
        ax.set_title(f"Veículo Inédito {vid} (~{vmed:.0f} V) — Viagem {tid}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tempo decorrido [minutos]")
        ax.set_ylabel("SOC [%]")
        ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    savefig(fig, "fig_ved_unseen_vehicles_temporal")
    plt.close(fig)

    # ==================================================================================
    # 6. Salvar JSON consolidado
    # ==================================================================================
    out = {
        "scenario_1_ved_temporal": c1,
        "scenario_2_ved_unseen_vehicles": {
            "split_summary": split_summary,
            "metrics": c2,
            "per_vehicle_compat": df_pv_c2.to_dict(orient="records"),
        },
        "scenario_1_per_vehicle_compat": df_pv_c1.to_dict(orient="records"),
        "scenario_3_external_jac": c3,
    }
    save_json(out, "intra_vs_external_comparison.json")
    print("\nExperimento comparativo salvo em results/intra_vs_external_comparison.json!")


if __name__ == "__main__":
    run()
