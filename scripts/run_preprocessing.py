"""
Executa o pré-processamento completo dos datasets conforme a metodologia do TCC (Seção 4.3).
Salva os datasets limpos, divididos e normalizados em data/processed/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.common import (
    DATA_PROCESSED, MODELS, RESULTS,
    TRAIN_COLS, TEST_COLS, FEATURES_FULL, FEATURES_COMPAT, TARGET, IQR_COLS,
    load_train_raw, clean_train, load_test_raw, clean_test,
    IQRClipper, add_temporal_features, temporal_split, save_json
)

def run():
    print("--- 1. Carregando dados brutos de treino (VED) ---")
    df_raw = load_train_raw()
    print(f"Shape bruto: {df_raw.shape}")

    print("\n--- 2. Limpeza de dados de treino ---")
    df_clean, clean_log = clean_train(df_raw, verbose=True)

    print("\n--- 3. Divisão temporal 70/15/15 respeitando viagens ---")
    df_clean["split"] = temporal_split(df_clean, fracs=(0.70, 0.15, 0.15))
    split_counts = df_clean["split"].value_counts()
    print("Contagem por split:")
    print(split_counts)
    print("Proporções:", (split_counts / len(df_clean)).round(3).to_dict())

    train_mask = df_clean["split"] == "train"
    val_mask = df_clean["split"] == "val"
    test_mask = df_clean["split"] == "test"

    print("\n--- 4. Tratamento de Outliers por IQR (ajustado apenas no treino) ---")
    clipper = IQRClipper(k=3.0)
    clipper.fit(df_clean[train_mask], cols=IQR_COLS)
    print("Limites IQR calculados:", clipper.bounds)

    df_clipped, outlier_counts = clipper.transform(df_clean, window=5)
    print("Outliers tratados por coluna:")
    print(outlier_counts)

    print("\n--- 5. Engenharia de atributos temporais (causais) ---")
    df_feat = add_temporal_features(df_clipped)
    print(f"Features disponíveis: {list(df_feat.columns)}")

    # Preenchimento de qualquer resíduo de elevação/gradiente após interpolação
    for col in ["elevation", "gradient"]:
        med_val = df_feat.loc[train_mask, col].median()
        df_feat[col] = df_feat[col].fillna(med_val)

    print("\n--- 6. Normalização Min-Max Scaling [0, 1] (ajustada no treino) ---")
    # Dois scalers: um para o conjunto FULL e um para o conjunto COMPAT
    scaler_full = MinMaxScaler(feature_range=(0, 1))
    scaler_full.fit(df_feat.loc[train_mask, FEATURES_FULL])

    scaler_compat = MinMaxScaler(feature_range=(0, 1))
    scaler_compat.fit(df_feat.loc[train_mask, FEATURES_COMPAT])

    joblib.dump(scaler_full, MODELS / "scaler_full.joblib")
    joblib.dump(scaler_compat, MODELS / "scaler_compat.joblib")
    joblib.dump(clipper, MODELS / "iqr_clipper.joblib")

    print("\n--- 7. Salvando datasets particionados ---")
    # Salvar splits brutos (com features temporais e clipped)
    train_df = df_feat[train_mask].reset_index(drop=True)
    val_df = df_feat[val_mask].reset_index(drop=True)
    test_df = df_feat[test_mask].reset_index(drop=True)

    train_df.to_parquet(DATA_PROCESSED / "train.parquet", index=False)
    val_df.to_parquet(DATA_PROCESSED / "val.parquet", index=False)
    test_df.to_parquet(DATA_PROCESSED / "test.parquet", index=False)
    print(f"Salvos: train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)})")

    print("\n--- 8. Carregando e processando dataset externo (JAC iEV40) ---")
    test_raw = load_test_raw()
    test_clean = clean_test(test_raw)
    test_feat = add_temporal_features(test_clean)
    test_feat.to_parquet(DATA_PROCESSED / "test_external.parquet", index=False)
    print(f"Dataset externo salvo: {test_feat.shape}")

    # Salvar metadados do pré-processamento
    prep_meta = {
        "n_raw": len(df_raw),
        "clean_log": clean_log,
        "split_counts": split_counts.to_dict(),
        "outlier_counts": outlier_counts.to_dict(),
        "features_full": FEATURES_FULL,
        "features_compat": FEATURES_COMPAT,
        "target": TARGET,
        "iqr_bounds": clipper.bounds
    }
    save_json(prep_meta, "preprocessing_metadata.json")
    print("\nPré-processamento concluído com sucesso!")

if __name__ == "__main__":
    run()
