# CP4 — Detecção de Fraude em Cartão de Crédito

Projeto acadêmico de Machine Learning para detectar fraude em transações de cartão de crédito usando MLP, rastreamento de experimentos e API local de predição.

Stack principal:

- Python 3.10+
- scikit-learn `MLPClassifier`
- imbalanced-learn
- MLflow
- FastAPI

Restrição importante: não usar TensorFlow, Keras ou PyTorch.

## Objetivo

Treinar, comparar e servir localmente um modelo MLP capaz de classificar uma nova transação como legítima ou fraude. Como o dataset é altamente desbalanceado, a seleção do modelo prioriza métricas da classe fraude: recall, precision, F1-score e ROC-AUC. Acurácia é reportada, mas não é usada como métrica principal.

## Estrutura

```text
mlp-deteccao-fraude/
├── data/raw/creditcard.csv      # dataset local, não versionado
├── models/                      # artifacts promovidos após treino
├── notebooks/eda.ipynb          # notebook obrigatório de EDA
├── reports/                     # resultados e roteiro do vídeo
├── scripts/test_api.py          # teste da API
├── src/                         # código de treino, preprocessamento e API
├── requirements.txt
└── README.md
```

## 1. Preparar ambiente

```bash
cd /Users/nando_mac/mlp-deteccao-fraude
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Baixar dataset

Dataset: Kaggle — `mlg-ulb/creditcardfraud`

Baixe o arquivo `creditcard.csv` e salve exatamente em:

```text
data/raw/creditcard.csv
```

O CSV bruto não deve ser commitado no Git.

Verificação:

```bash
test -f data/raw/creditcard.csv && echo "dataset ok"
```

### Configurar seu token Kaggle (necessário só para baixar o dataset)

O download via Kaggle (site ou CLI/API) exige um token pessoal. Esse token é **seu**, não é compartilhado no repositório e nunca deve ser commitado.

1. Crie uma conta no Kaggle (se ainda não tiver).
2. Acesse **Account → API → Create New Token**. Isso baixa um arquivo `kaggle.json`.
3. Salve esse arquivo **fora do projeto**, em `~/.kaggle/kaggle.json`, e restrinja a permissão:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. Baixe o dataset com a CLI oficial (ou baixe manualmente pela página do dataset no navegador, sem usar token nenhum):

   ```bash
   pip install kaggle
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
   ```

Cada pessoa do grupo usa o próprio token — nunca copie ou envie o `kaggle.json` de outra pessoa, e nunca coloque esse arquivo dentro da pasta do projeto (o `.gitignore` já bloqueia `kaggle.json` e `.kaggle/` como proteção extra, mas a regra principal é: o token não deve nem existir dentro do repositório).

## 3. Rodar o notebook obrigatório de EDA

```bash
jupyter notebook notebooks/eda.ipynb
```

O notebook documenta:

- tamanho do dataset;
- colunas disponíveis;
- valores ausentes;
- distribuição da classe `Class`;
- percentual de fraude;
- estatísticas e gráficos de `Time` e `Amount`.

![Distribuição da classe (legítima vs fraude)](docs/images/notebook-class-distribution.png)

## 4. Treinar modelos e registrar no MLflow

```bash
python -m src.train
```

O treinamento executa cinco experimentos MLP definidos em `src/experiments.py`, com split estratificado treino/validação/teste. SMOTE ou undersampling é aplicado somente no treino.

Saídas esperadas:

```text
reports/experiment_results.csv
reports/model_selection.md
models/best_model.joblib
models/preprocessor.joblib
models/best_run.json
mlruns/
```

## 5. Comparar runs no MLflow

```bash
mlflow ui
```

Abra a URL indicada pelo terminal e compare os runs por parâmetros e métricas.

Métricas registradas:

- accuracy;
- precision da classe fraude;
- recall da classe fraude;
- F1-score da classe fraude;
- ROC-AUC.

![Lista de runs no MLflow, MLP-01 a MLP-05](docs/images/mlflow-runs-list.png)

![Comparação de runs no MLflow, hiperparâmetros vs métricas](docs/images/mlflow-runs-compare.png)

## 6. Critério de seleção do melhor modelo

O melhor modelo é escolhido pelo maior F1-score da classe fraude entre os modelos com recall aceitável. Em caso de empate, ROC-AUC desempata.

Resposta esperada no README/vídeo após rodar o treino:

> O modelo selecionado foi o indicado em `reports/model_selection.md`. Ele foi escolhido por apresentar o melhor equilíbrio entre recall e precision da classe fraude, maximizando o F1-score entre os experimentos com recall aceitável. O ROC-AUC foi usado como critério de desempate. As métricas finais em teste estão registradas em `models/best_run.json` e no relatório de seleção.

Se alguma meta de referência não for atingida, documente o tradeoff honestamente. O projeto continua defensável se todos os experimentos estiverem rastreados e a seleção estiver bem justificada.

Metas de referência:

- recall fraude >= 0.80;
- precision fraude >= 0.60;
- F1-score fraude >= 0.70;
- ROC-AUC >= 0.90.

## 7. Iniciar API local

Depois do treino gerar os artifacts em `models/`:

```bash
uvicorn src.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

A documentação interativa fica disponível em `http://127.0.0.1:8000/docs`:

![Swagger UI do FastAPI mostrando POST /predict](docs/images/api-swagger-docs.png)

## 8. Testar predição

Em outro terminal, com o ambiente ativado:

```bash
python scripts/test_api.py
```

Resposta esperada:

```json
{
  "prediction": 0,
  "label": "legitima",
  "fraud_probability": 0.03
}
```

Os valores podem mudar conforme o modelo treinado.

![Terminal com a resposta de python scripts/test_api.py](docs/images/api-test-response.png)

## Contrato POST /predict

Entrada JSON com todos os campos numéricos:

```json
{
  "Time": 10.2,
  "V1": -1.35,
  "V2": -0.07,
  "V3": 2.53,
  "V4": 1.38,
  "V5": -0.34,
  "V6": 0.46,
  "V7": 0.24,
  "V8": 0.10,
  "V9": 0.36,
  "V10": 0.09,
  "V11": -0.55,
  "V12": -0.62,
  "V13": -0.99,
  "V14": -0.31,
  "V15": 1.47,
  "V16": -0.47,
  "V17": 0.20,
  "V18": 0.03,
  "V19": 0.40,
  "V20": 0.25,
  "V21": -0.02,
  "V22": 0.28,
  "V23": -0.11,
  "V24": 0.07,
  "V25": 0.13,
  "V26": -0.19,
  "V27": 0.13,
  "V28": -0.02,
  "Amount": 149.62
}
```

Saída:

```json
{
  "prediction": 0,
  "label": "legitima",
  "fraud_probability": 0.03
}
```

## Troubleshooting

### Dataset não encontrado

Erro esperado se `data/raw/creditcard.csv` estiver ausente. Baixe do Kaggle e coloque no caminho correto.

### API retorna erro 503

Significa que os artifacts do modelo ainda não existem. Rode antes:

```bash
python -m src.train
```

### MLflow UI não mostra runs

Confirme que o comando foi executado a partir da raiz do projeto e que a pasta `mlruns/` foi criada localmente.

## Prints do README

Os 5 prints usados neste README estão salvos em `docs/images/`:

| Arquivo | Conteúdo |
| --- | --- |
| `docs/images/notebook-class-distribution.png` | Distribuição da classe `Class` no dataset. |
| `docs/images/mlflow-runs-list.png` | Tela do MLflow listando os 5 runs `MLP-01` a `MLP-05`. |
| `docs/images/mlflow-runs-compare.png` | Comparação visual das métricas dos 5 experimentos. |
| `docs/images/api-swagger-docs.png` | Swagger UI do FastAPI com endpoint `POST /predict`. |
| `docs/images/api-test-response.png` | Resposta do comando `python scripts/test_api.py`. |

## Vídeo

Use o roteiro em:

```text
https://drive.google.com/file/d/1mbYVa6JqWQfUCe69Pbrn0bDp3qqj31zE/view?usp=sharing
```

