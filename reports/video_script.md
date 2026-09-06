## 0:00 – 0:20 | Objetivo

Nosso projeto busca detectar fraudes em transações de cartão de crédito usando uma rede neural MLP. A ideia não foi só treinar um modelo: nós também comparamos diferentes configurações com o MLflow, selecionamos o melhor experimento e, no final, disponibilizamos esse modelo em uma API local para fazer novas predições.

---

## 0:20 – 1:10 | Dataset e preparação

Para o projeto, escolhemos o dataset Credit Card Fraud Detection, que é diferente da base de Breast Cancer utilizada em aula.

Antes do treinamento, encontramos e removemos 1.081 registros exatamente duplicados, mantendo o arquivo original preservado. Depois dessa limpeza, ficamos com 283.726 transações, sendo apenas 473 fraudes — cerca de 0,17% da base.

Esse desbalanceamento é importante porque, nesse cenário, olhar só para acurácia seria enganoso. Um modelo que classificasse tudo como transação legítima teria uma acurácia muito alta, mas não detectaria nenhuma fraude.

Por isso, dividimos os dados em 70% para treino, 15% para validação e 15% para teste, garantindo que não existissem registros idênticos entre esses conjuntos. O scaler foi ajustado somente com os dados de treino, e as estratégias de balanceamento, como SMOTE e undersampling, também foram aplicadas exclusivamente nessa etapa.

---

## 1:10 – 2:00 | MLP e hiperparâmetros

O modelo utilizado foi o MLPClassifier do scikit-learn, seguindo a orientação de não utilizar TensorFlow, Keras ou PyTorch.

A entrada da rede possui 30 características da transação. A partir delas, o MLP aprende padrões que ajudam a diferenciar uma operação legítima de uma possível fraude.

Nós criamos cinco experimentos, variando a arquitetura da rede, quantidade de neurônios, taxa de aprendizado, função de ativação e estratégia de balanceamento. Testamos, por exemplo, redes com uma camada de 32 neurônios e redes maiores, com duas camadas de 64 e 32 ou 128 e 64 neurônios.

O treinamento utiliza o otimizador Adam e early stopping para evitar continuar treinando quando o modelo deixa de melhorar. Além disso, o limiar de classificação também foi escolhido usando somente a validação, sem consultar o conjunto de teste durante a seleção.

---

## 2:00 – 3:25 | MLflow e escolha do modelo

Para acompanhar os experimentos, utilizamos o MLflow. Cada execução registra os hiperparâmetros utilizados, as métricas de validação e os artefatos necessários para reproduzir o modelo.

Como nosso dataset é muito desbalanceado, demos mais atenção a precision, recall e F1 do que à acurácia.

A precision mostra quantos dos alertas gerados realmente eram fraude. Já o recall mostra quantas das fraudes existentes o modelo conseguiu encontrar. E o F1 ajuda a avaliar o equilíbrio entre essas duas métricas.

Antes dos experimentos, definimos uma regra de seleção: primeiro o modelo precisava alcançar pelo menos 80% de recall na validação. Entre os modelos que cumprissem esse requisito, escolheríamos aquele com maior F1, usando ROC-AUC como desempate.

Os cinco modelos finais atingiram o recall mínimo. O melhor resultado foi o MLP-03, com 76% de precision, 80,28% de recall e 78,08% de F1 na validação. O limiar selecionado para esse modelo foi 0,83.

A escolha termina aqui, na validação. O conjunto de teste ficou separado justamente para avaliarmos depois como o modelo escolhido se comportaria em dados que não participaram dessa decisão.

---

## 3:25 – 4:35 | API e predição

Depois de selecionar o vencedor, integramos o modelo com uma API local.

A API recupera diretamente do MLflow o modelo correspondente ao run selecionado, além do preprocessador e do limiar utilizados naquele mesmo experimento. Isso garante que uma nova transação passe exatamente pela mesma preparação utilizada durante o treinamento.

Aqui no Swagger podemos enviar as 30 características de uma transação para o endpoint de predição.

Ao executar a requisição, recebemos uma resposta HTTP 200 contendo a classe prevista, o rótulo e a probabilidade estimada de fraude.

Essa probabilidade é comparada com o limiar de 0,83 escolhido anteriormente na validação para definir a classificação final da transação.

---

## 4:35 – 5:00 | Resultado e conclusão

Por fim, avaliamos o MLP-03 no conjunto de teste.

Ele obteve 66,67% de precision, 78,87% de recall e 72,26% de F1. Das 71 fraudes presentes no teste, o modelo identificou 56 e deixou de detectar 15, além de gerar 28 falsos positivos.

O recall ficou um pouco abaixo dos 80% que usamos como referência na validação, e nós mantivemos esse resultado sem reajustar o modelo com base no teste.

Com isso, concluímos o projeto com o MLP treinado, os experimentos rastreados no MLflow, uma seleção baseada em critérios definidos previamente e o modelo final disponível através de uma API local.
