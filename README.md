# Predição do Estado de Carga (SOC) em Baterias de Veículos Elétricos com Random Forest e XGBoost

Repositório oficial com o código-fonte utilizado nos experimentos do Trabalho de Conclusão de Curso (TCC) em **Engenharia da Computação** (UEMG Unidade Divinópolis):

> **Título:** *Predição do Estado de Carga em Baterias de Veículos Elétricos a partir de Dados CAN: uma Abordagem com Random Forest e XGBoost*  
> **Autor:** Felipe Alves Costa  
> **Orientação:** Prof. Me. Marcos Alberto Saldanha | **Coorientação:** Prof. Dr. Arismar Morais Gonçalves Júnior

Este repositório contém **apenas o código necessário para reproduzir todo o pipeline experimental**: pré-processamento, criação de atributos causais, análise exploratória, treinamento e otimização de hiperparâmetros (*Random Forest* e XGBoost), validação interna (temporal e por veículos inéditos), validação externa em veículo distinto (JAC iEV40) e benchmark de inferência em CPU.

---

## Estrutura do Projeto

```text
tcc-soc-prediction/
├── data/
│   └── raw/                             # Coloque aqui os 2 arquivos CSV baixados do Kaggle
├── src/
│   └── common.py                        # Limpeza, IQR causal, atributos derivados, splits e métricas
├── scripts/
│   ├── run_preprocessing.py             # Limpeza, divisão 70/15/15, IQR, atributos causais e scalers
│   ├── generate_eda_figures.py          # Figuras da análise exploratória (EDA)
│   ├── train_random_forest.py           # Grid Search (TimeSeriesSplit) + RF FULL e COMPAT
│   ├── train_xgboost.py                 # Otimização Bayesiana (Optuna) + XGBoost FULL e COMPAT
│   ├── evaluate_external_validation.py  # Inferência e diagnóstico por sessão no JAC iEV40 (CAN)
│   ├── compare_intra_vs_external.py     # Comparação Cenário 1 vs. Cenário 2 (veículos inéditos) vs. Cenário 3
│   └── benchmark_inference.py           # Medição de latência de inferência em CPU (single-thread)
├── run_all.py                           # Executa todas as etapas em sequência
├── requirements.txt                     # Dependências Python com versões exatas
└── LICENSE
```

Durante a execução, os diretórios `data/processed/`, `models/`, `results/` e `figures/` são criados automaticamente.

---

## Requisitos

- **Python:** 3.12 (testado com Python 3.12.10)
- **Memória RAM recomendada:** 4 GB ou mais (o arquivo bruto do VED possui 5 milhões de linhas, ~910 MB)
- **Espaço em disco:** ~1,5 GB livres (CSVs brutos + arquivos `.parquet` intermediários + modelos `.joblib`)

---

## 1. Download dos Conjuntos de Dados

Os arquivos CSV não são versionados neste repositório devido ao tamanho e à licença das bases originais. Baixe ambos os conjuntos públicos no Kaggle:

1. **Base de Treinamento e Validação Interna (VED):**  
   - **Nome:** *Vehicle Energy & Telemetry Dataset* (Souza, 2025; baseado em Oh et al., 2022)  
   - **Link:** https://www.kaggle.com/dsv/14190605 (DOI: `10.34740/KAGGLE/DSV/14190605`)  
   - **Arquivo esperado:** `data/raw/vehicle_data.csv`

2. **Base de Validação Externa (JAC iEV40 via CAN):**  
   - **Nome:** *Electric Vehicle Telemetry Dataset* (Reis, 2025)  
   - **Link:** https://www.kaggle.com/datasets/marceloreis76/electric-vehicle-telemetry-dataset  
   - **Arquivo esperado:** `data/raw/dataset.csv`

> **Caminhos personalizados (opcional):** se os arquivos já estiverem em outra pasta do seu computador, você pode informar os caminhos por variáveis de ambiente sem mover os CSVs:
> - `TCC_TRAIN_CSV`: caminho completo para `vehicle_data.csv`
> - `TCC_TEST_CSV`: caminho completo para `dataset.csv`

---

## 2. Instalação do Ambiente

Clone o repositório e crie um ambiente virtual isolado:

### Windows (PowerShell)

```powershell
git clone https://github.com/SEU_USUARIO/tcc-soc-prediction.git
cd tcc-soc-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone https://github.com/SEU_USUARIO/tcc-soc-prediction.git
cd tcc-soc-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Como Executar o Experimento

### Opção A: Pipeline Completo em Um Comando

Com os dois CSVs em `data/raw/` (ou configurados via `TCC_TRAIN_CSV` e `TCC_TEST_CSV`), execute:

```bash
python run_all.py
```

Parâmetros opcionais:
- `python run_all.py --skip-eda`: pula a geração dos gráficos de análise exploratória.
- `python run_all.py --skip-benchmark`: pula o benchmark de latência de inferência.

### Opção B: Etapa por Etapa

```bash
# 1. Limpeza, divisão temporal 70/15/15, IQR, atributos causais e normalização Min-Max
python scripts/run_preprocessing.py

# 2. Figuras de análise exploratória (EDA)
python scripts/generate_eda_figures.py

# 3. Treinamento e avaliação do Random Forest (FULL e COMPAT)
python scripts/train_random_forest.py

# 4. Otimização Bayesiana (Optuna) e treinamento do XGBoost (FULL e COMPAT)
python scripts/train_xgboost.py

# 5. Validação externa direta no JAC iEV40 (sessões COND 1 a 9)
python scripts/evaluate_external_validation.py

# 6. Experimento comparativo: Cenário 1 (Temporal) vs. Cenário 2 (Veículos Inéditos) vs. Cenário 3 (JAC)
python scripts/compare_intra_vs_external.py

# 7. Benchmark de latência de inferência em CPU (thread única)
python scripts/benchmark_inference.py
```

---

## 4. Saídas Geradas e Resultados de Referência

Ao final da execução, serão gerados:
- `results/*.json`: todas as métricas consolidadas (RMSE, MAE em pontos percentuais de SOC, $R^2$, intervalos de confiança de 95% via *bootstrap* em blocos e tempos de inferência);
- `figures/*.png` e `figures/*.pdf`: gráficos de EDA, importância de atributos, resíduos, curvas temporais e comparações entre cenários;
- `models/*.joblib`: modelos treinados (*Random Forest* e XGBoost, configurações `FULL` com 15 atributos e `COMPAT` com 10 atributos), `MinMaxScaler` e `IQRClipper`.

### Resultados de Referência Obtidos no TCC (semente fixa `SEED = 42`)

| Cenário de Avaliação | Conjunto | Modelo | RMSE [p.p.] | MAE [p.p.] | R² |
| :--- | :---: | :--- | :---: | :---: | :---: |
| 1. VED Temporal (mesmos veículos) | `FULL` (15) | Random Forest | 10,86 | 7,25 | 0,8762 |
| 1. VED Temporal (mesmos veículos) | `FULL` (15) | **XGBoost** | **10,29** | **7,00** | **0,8889** |
| 1. VED Temporal (mesmos veículos) | `COMPAT` (10) | Random Forest | 10,27 | **7,30** | 0,8891 |
| 1. VED Temporal (mesmos veículos) | `COMPAT` (10) | **XGBoost** | **10,23** | 7,48 | **0,8901** |
| 2. VED Veículos Inéditos (0% sobreposição) | `FULL` (15) | Random Forest | 19,60 | 13,87 | 0,5561 |
| 2. VED Veículos Inéditos (0% sobreposição) | `FULL` (15) | **XGBoost** | **18,25** | **13,18** | **0,6151** |
| 2. VED Veículos Inéditos (0% sobreposição) | `COMPAT` (10) | Random Forest | 18,85 | 13,95 | 0,5894 |
| 2. VED Veículos Inéditos (0% sobreposição) | `COMPAT` (10) | **XGBoost** | **18,20** | **13,55** | **0,6174** |
| 3a. Externo JAC iEV40 (Sessão 7, ~362 V) | `COMPAT` (10) | Random Forest | 22,03 | **20,44** | -2,0568 |
| 3a. Externo JAC iEV40 (Sessão 7, ~362 V) | `COMPAT` (10) | **XGBoost** | **22,00** | 20,81 | **-2,0458** |
| 3b. Externo JAC iEV40 (Sessões válidas 7 e 8) | `COMPAT` (10) | Random Forest | 31,99 | 28,17 | -2,2205 |
| 3b. Externo JAC iEV40 (Sessões válidas 7 e 8) | `COMPAT` (10) | **XGBoost** | **31,04** | **27,71** | **-2,0330** |
| 3c. Externo JAC iEV40 (Global, inclui sessões 0 a 25 V) | `COMPAT` (10) | Random Forest | 60,04 | 54,57 | -7,9862 |
| 3c. Externo JAC iEV40 (Global, inclui sessões 0 a 25 V) | `COMPAT` (10) | **XGBoost** | **58,92** | **53,37** | **-7,6520** |

> **Observação sobre reprodutibilidade:** pequenas variações na segunda casa decimal podem ocorrer caso sejam utilizadas versões diferentes de `scikit-learn`, `xgboost` ou `optuna`, ou outro número de threads no treinamento paralelo.

---

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
