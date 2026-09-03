# Roteiro do Vídeo — CP4 Detecção de Fraude em Cartão de Crédito

Tempo máximo: 5 minutos. Gravar com um único integrante apresentando a tela.

## 0:00–0:30 — Problema

Apresentar o objetivo: prever se uma transação de cartão é legítima ou fraude. Destacar que o dataset é muito desbalanceado, com fraudes representando menos de 0,2% das transações, então acurácia isolada não é suficiente.

## 0:30–1:15 — Notebook obrigatório de EDA

Abrir `notebooks/eda.ipynb` e mostrar:

- shape do dataset;
- colunas `Time`, `Amount`, `V1`–`V28` e `Class`;
- ausência de nulos;
- distribuição da classe;
- estatísticas e gráficos simples de `Time` e `Amount`.

## 1:15–2:15 — Treinamento MLP

Mostrar `src/experiments.py` e explicar que foram definidos cinco experimentos com variações de arquitetura, learning rate, ativação e estratégia de balanceamento. Reforçar que o MLP usa `MLPClassifier` do scikit-learn, sem TensorFlow/Keras.

## 2:15–3:30 — MLflow e seleção

Abrir a interface do MLflow com `mlflow ui`. Mostrar os runs, parâmetros e métricas. Explicar que a escolha prioriza F1-score da classe fraude com recall aceitável, usando ROC-AUC como desempate. Abrir `reports/model_selection.md` para mostrar a justificativa.

## 3:30–4:30 — API local

Iniciar a API com:

```bash
uvicorn src.api:app --reload
```

Em outro terminal, rodar:

```bash
python scripts/test_api.py
```

Mostrar a resposta com `prediction`, `label` e `fraud_probability`.

## 4:30–5:00 — Conclusão

Dizer qual modelo venceu e por quê, citando recall, precision, F1-score e ROC-AUC. Se alguma meta não foi atingida, explicar honestamente o tradeoff observado e por que o modelo ainda é o melhor da matriz testada.
