"""
Módulo compartilhado do TCC: Predição do SOC com Random Forest e XGBoost.

Concentra caminhos, carregamento, limpeza, engenharia de atributos, divisão temporal,
métricas e utilitários gráficos, garantindo que todos os scripts utilizem exatamente
o mesmo pré-processamento.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Caminhos (relativos à raiz do repositório; sobrescrevíveis por variáveis de ambiente)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
for _p in (DATA_RAW, DATA_PROCESSED, FIGURES, MODELS, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path(os.environ.get("TCC_TRAIN_CSV", DATA_RAW / "vehicle_data.csv"))
TEST_CSV = Path(os.environ.get("TCC_TEST_CSV", DATA_RAW / "dataset.csv"))

SEED = 42

# --------------------------------------------------------------------------------------
# Mapeamento de colunas
# --------------------------------------------------------------------------------------
TRAIN_COLS = {
    "DayNum": "day",
    "VehId": "veh",
    "Trip": "trip",
    "Timestamp(ms)": "ts_ms",
    "Vehicle Speed[km/h]": "speed",
    "OAT[DegC]": "oat",
    "Air Conditioning Power[kW]": "ac_kw",
    "Air Conditioning Power[Watts]": "ac_w",
    "Heater Power[Watts]": "heater_w",
    "HV Battery Current[A]": "current",
    "HV Battery SOC[%]": "soc",
    "HV Battery Voltage[V]": "voltage",
    "Elevation Smoothed[m]": "elevation",
    "Gradient": "gradient",
    "Energy_Consumption": "energy_cons",
}

# Dataset externo (JAC iEV40) -> nomes padronizados
TEST_COLS = {
    "CH": "soc", "VOL": "voltage", "CUR": "current", "SPD": "speed", "AIR": "air",
    "AUT": "autonomy", "ECO": "eco", "BRK": "brake", "ACC": "accel_pedal",
    "ALT": "altitude", "COND": "cond", "ODO": "odo",
}

# Rótulos legíveis para gráficos
LABELS = {
    "speed": "Velocidade [km/h]", "current": "Corrente HV [A]", "voltage": "Tensão HV [V]",
    "oat": "Temp. externa [°C]", "ac_power_w": "Potência A/C [W]", "heater_w": "Potência aquecedor [W]",
    "elevation": "Elevação [m]", "gradient": "Gradiente", "energy_cons": "Consumo energético",
    "power_kw": "Potência HV [kW]", "current_ma5": "Corrente (média 5)", "current_ma20": "Corrente (média 20)",
    "voltage_ma5": "Tensão (média 5)", "voltage_ma20": "Tensão (média 20)", "speed_ma5": "Velocidade (média 5)",
    "ac_on": "A/C ligado", "soc": "SOC [%]",
}

# --------------------------------------------------------------------------------------
# Conjuntos de atributos
# --------------------------------------------------------------------------------------
ENGINEERED = ["power_kw", "current_ma5", "current_ma20", "voltage_ma5", "voltage_ma20", "speed_ma5"]

# Configuração A: todas as variáveis da base VED (+ atributos temporais derivados) -> 15 atributos
FEATURES_FULL = [
    "speed", "current", "voltage", "oat", "ac_power_w", "heater_w",
    "elevation", "gradient", "energy_cons",
] + ENGINEERED

# Configuração B: apenas variáveis compatíveis com as duas bases (VED e JAC iEV40) -> 10 atributos
FEATURES_COMPAT = ["speed", "current", "voltage", "ac_on"] + ENGINEERED

FEATURE_SETS = {"full": FEATURES_FULL, "compat": FEATURES_COMPAT}
TARGET = "soc"

# Variáveis contínuas sujeitas ao tratamento de outliers por IQR
IQR_COLS = ["speed", "current", "voltage", "oat", "ac_power_w", "heater_w", "elevation", "gradient"]
IQR_K = 3.0  # cerca "extrema" de Tukey: preserva picos reais de corrente (aceleração/regeneração)


# --------------------------------------------------------------------------------------
# Carregamento e limpeza
# --------------------------------------------------------------------------------------
def load_train_raw() -> pd.DataFrame:
    """Lê as colunas relevantes do dataset de treinamento (VED)."""
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo de treinamento não encontrado em: {TRAIN_CSV}\n"
            "Baixe o Vehicle Energy & Telemetry Dataset (vehicle_data.csv) conforme o README.md "
            "e coloque-o em data/raw/vehicle_data.csv (ou defina a variável TCC_TRAIN_CSV)."
        )
    df = pd.read_csv(TRAIN_CSV, usecols=list(TRAIN_COLS))
    return df.rename(columns=TRAIN_COLS)


def clean_train(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Limpeza do dataset de treinamento. Retorna (df_limpo, log_de_etapas)."""
    log = {"linhas_brutas": len(df)}

    # 1) Linhas sem SOC/tensão/corrente -> veículos a combustão (ICE/HEV sem sinal HV): removidas
    df = df.dropna(subset=["soc", "voltage", "current"])
    log["apos_remover_nulos_chave"] = len(df)

    # 2) SOC == 0 com tensão nominal -> PHEV em modo "charge-sustaining" (SOC reportado é o SOC útil
    #    do modo elétrico, não o estado eletroquímico). Também removemos tensão nula (falha de leitura).
    df = df[(df["soc"] > 0) & (df["voltage"] > 0)]
    log["apos_remover_soc0_vol0"] = len(df)

    # 3) Identificador único de viagem (Trip se repete entre veículos)
    df = df.copy()
    df["trip_id"] = (df["veh"].astype(int) * 100_000 + df["trip"].astype(int)).astype(np.int64)

    # 4) Ordenação temporal: dia -> viagem -> timestamp (o CSV original vem embaralhado)
    df = df.sort_values(["day", "trip_id", "ts_ms"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["trip_id", "ts_ms"])
    log["apos_remover_duplicatas"] = len(df)

    # 5) Potência do A/C: o dataset tem duas colunas complementares (kW e W)
    df["ac_power_w"] = df["ac_w"].fillna(df["ac_kw"] * 1000.0)
    df["ac_power_w"] = df["ac_power_w"].fillna(0.0)
    df["ac_on"] = (df["ac_power_w"] > 0).astype(np.int8)

    # 6) Aquecedor ausente = veículo não reporta o sinal -> 0 W
    df["heater_w"] = df["heater_w"].fillna(0.0)

    # 7) Elevação/gradiente ausentes: interpolação dentro da viagem; resto preenchido depois (mediana do treino)
    for c in ["elevation", "gradient"]:
        df[c] = df.groupby("trip_id")[c].transform(lambda s: s.interpolate(limit_direction="both"))

    # 8) Energy_Consumption é linear em V·I (corr = 1,0). Reconstrói valores ausentes pela relação exata.
    p = df["voltage"] * df["current"]
    mask = df["energy_cons"].notna()
    coef = np.polyfit(p[mask], df.loc[mask, "energy_cons"], 1)
    df.loc[~mask, "energy_cons"] = np.polyval(coef, p[~mask])
    log["energy_cons_coef"] = [float(c) for c in coef]

    df = df.drop(columns=["ac_kw", "ac_w"])
    if verbose:
        for k, v in log.items():
            print(f"{k:>28}: {v:,}" if isinstance(v, int) else f"{k:>28}: {v}")
    return df, log


def load_test_raw() -> pd.DataFrame:
    """Lê o dataset de validação externa (JAC iEV40)."""
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo de validação externa não encontrado em: {TEST_CSV}\n"
            "Baixe o Electric Vehicle Telemetry Dataset (dataset.csv) conforme o README.md "
            "e coloque-o em data/raw/dataset.csv (ou defina a variável TCC_TEST_CSV)."
        )
    return pd.read_csv(TEST_CSV)


def clean_test(raw: pd.DataFrame) -> pd.DataFrame:
    """Harmoniza o dataset CAN do JAC iEV40 com o padrão do dataset de treinamento."""
    df = raw.sort_values("id").rename(columns=TEST_COLS).reset_index(drop=True)
    # Convenção de sinal: no VED a corrente de DESCARGA é negativa; no JAC é positiva
    # (correlação CUR x pedal do acelerador = +0,91). Invertemos para alinhar.
    df["current"] = -df["current"].astype(float)
    df["voltage"] = df["voltage"].astype(float)
    df["speed"] = df["speed"].astype(float)
    df["ac_on"] = (df["air"] > 0).astype(np.int8)
    df["trip_id"] = df["cond"].astype(int)
    # Tensão válida: pack de ~350 V. Valores < 250 V (sessões 1-6 e 9, entre 0 e 25 V) são incompatíveis com tração.
    df["voltage_valid"] = df["voltage"] >= 250
    df["seq"] = df.groupby("trip_id").cumcount()
    return df


# --------------------------------------------------------------------------------------
# Outliers (IQR): limites ajustados apenas no treino
# --------------------------------------------------------------------------------------
@dataclass
class IQRClipper:
    k: float = IQR_K
    bounds: dict | None = None

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "IQRClipper":
        self.bounds = {}
        for c in cols:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:  # variável majoritariamente constante (ex.: gradiente, aquecedor): não trata
                continue
            self.bounds[c] = (float(q1 - self.k * iqr), float(q3 + self.k * iqr))
        return self

    def transform(self, df: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, pd.Series]:
        """Substitui outliers pela mediana móvel CAUSAL (janela temporal) da própria viagem."""
        df = df.copy()
        counts = {}
        for c, (lo, hi) in self.bounds.items():
            if c not in df:
                continue
            out = (df[c] < lo) | (df[c] > hi)
            counts[c] = int(out.sum())
            if out.any():
                clean = df[c].where(~out)
                med = clean.groupby(df["trip_id"]).transform(
                    lambda s: s.rolling(window, min_periods=1).median().ffill().bfill()
                )
                df.loc[out, c] = med[out].fillna(np.clip(df.loc[out, c], lo, hi))
        return df, pd.Series(counts, name="outliers_tratados")


# --------------------------------------------------------------------------------------
# Engenharia de atributos (causal: só usa amostras passadas -> viável em sistema embarcado)
# --------------------------------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("trip_id", sort=False)
    df["power_kw"] = df["voltage"] * df["current"] / 1000.0
    df["current_ma5"] = g["current"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["current_ma20"] = g["current"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    df["voltage_ma5"] = g["voltage"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["voltage_ma20"] = g["voltage"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    df["speed_ma5"] = g["speed"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    return df


# --------------------------------------------------------------------------------------
# Divisão temporal 70/15/15 respeitando fronteiras de viagem
# --------------------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, fracs=(0.70, 0.15, 0.15)) -> pd.Series:
    """Retorna série 'split' em {train, val, test}. Viagens inteiras, ordenadas pelo dia."""
    trip_order = df.groupby("trip_id")["day"].min().sort_values()
    sizes = df.groupby("trip_id").size().loc[trip_order.index]
    cum = sizes.cumsum() / sizes.sum()
    labels = np.where(cum <= fracs[0], "train", np.where(cum <= fracs[0] + fracs[1], "val", "test"))
    mapping = pd.Series(labels, index=trip_order.index)
    return df["trip_id"].map(mapping)


# --------------------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------------------
def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": int(len(y_true))}


def bootstrap_ci(y_true, y_pred, groups, n_boot: int = 500, seed: int = SEED) -> dict:
    """IC 95% por bootstrap em blocos (viagens), preservando a dependência temporal."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"y": y_true, "p": y_pred, "g": groups})
    agg = d.assign(se=(d.p - d.y) ** 2, ae=(d.p - d.y).abs()).groupby("g").agg(
        se=("se", "sum"), ae=("ae", "sum"), n=("y", "size")
    )
    ids = agg.index.to_numpy()
    rm, ma = [], []
    for _ in range(n_boot):
        s = agg.loc[rng.choice(ids, size=len(ids), replace=True)]
        rm.append(np.sqrt(s.se.sum() / s.n.sum()))
        ma.append(s.ae.sum() / s.n.sum())
    return {
        "RMSE_ci": (float(np.percentile(rm, 2.5)), float(np.percentile(rm, 97.5))),
        "MAE_ci": (float(np.percentile(ma, 2.5)), float(np.percentile(ma, 97.5))),
    }


def save_json(obj, name: str):
    with open(RESULTS / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_json(name: str):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Estilo gráfico
# --------------------------------------------------------------------------------------
PALETTE = {"rf": "#2E7D32", "xgb": "#C62828", "real": "#263238", "lin": "#1565C0", "mean": "#9E9E9E"}


def set_style():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    })


def savefig(fig, name: str):
    fig.savefig(FIGURES / f"{name}.png")
    fig.savefig(FIGURES / f"{name}.pdf")
