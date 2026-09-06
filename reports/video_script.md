Roteiro de gravação

MLP + MLflow + API local de predição

Duração planejada: até 5 minutos. As marcações incluem fala, troca de telas e demonstração. Ensaiar com cronômetro; o texto não garante a duração final.

Antes de gravar - notas para o grupo

A restrição a TensorFlow, Keras e PyTorch foi uma orientação do professor em sala. O projeto utiliza MLPClassifier do scikit-learn e atende a essa orientação.

Duplicatas resolvidas: removemos 1.081 registros exatamente repetidos antes do split. Restaram 283.726 transações e 473 fraudes. O CSV original foi preservado. A verificação confirmou zero entradas idênticas entre treino, validação e teste; os cinco experimentos foram refeitos.

Resultados finais sem duplicatas: MLP-03, camadas (64, 32), SMOTE, ReLU, Adam, taxa de aprendizado 0,0001 e limiar 0,83. Validação: precision 0,7600, recall 0,8028, F1 0,7808 e ROC-AUC 0,9617. Teste: precision 0,6667, recall 0,7887, F1 0,7226 e ROC-AUC 0,9535. O recall de teste ficou abaixo da referência de 80%; não omitir esse resultado.

Os cinco novos runs foram registrados com dataset_policy=exact_duplicates_removed_v1. Vencedor: MLP-03, run 80642936d52a4121858484fb3acb1877. Métricas finais no MLflow com prefixo test_*; selected_model=true identifica o vencedor atual. Os runs anteriores foram preservados como histórico.

A API recupera o modelo pelo endereço runs:/<run_id>/sklearn_model no MLflow local, e carrega o preprocessador e o limiar do mesmo run. A integração foi validada por testes e por uma requisição HTTP com resposta 200. Manter mlruns/ e models/best_run.json disponíveis.

Deixar pronto na tela

README, notebook executado e src/experiments.py abertos.

MLflow aberto com o filtro tags.dataset_policy = 'exact_duplicates_removed_v1', mostrando apenas os cinco runs corrigidos. Para esta execução específica, usar também tags.training_batch = 'd3dc94a21c824076b64e7759b5935dbb'.

Relatório de seleção final e identificação do run vencedor visíveis.

API iniciada, /health indicando ready e Swagger em http://127.0.0.1:8000/docs.

JSON válido já separado para colar em POST /predict. Ensaiar a requisição e conferir a resposta.

Usar novas capturas de MLflow e API. Os Takes anteriores à deduplicação mostram outra avaliação e não devem ser usados para apresentar os resultados atuais.

As notas desta página não são falas. As próximas páginas contém a sequência de apresentação.

0:00 - 0:20 | Objetivo

Mostrar

README: título e objetivo. Evitar percorrer toda a documentação.

Fala sugerida

Nosso projeto usa uma rede neural MLP para classificar transações de cartão como legítimas ou fraudulentas. Além de treinar o modelo, registramos os experimentos no MLflow, comparamos os resultados e disponibilizamos uma API local de predição.

0:20 - 1:10 | Dataset e preparação

Mostrar

Notebook: tamanho, contagem de Class e gráfico de distribuição. Em seguida, apontar a preparação em src/train.py.

Fala sugerida

Escolhemos o Credit Card Fraud Detection, diferente do Breast Cancer da aula. Removemos 1.081 registros repetidos antes da divisão, preservando a base original. Restaram 283.726 transações, com 473 fraudes: cerca de 0,17%. Acurácia sozinha não basta, porque prever sempre legítima deixaria passar todas as fraudes. Dividimos a base em 70% para treino, 15% para validação e 15% para teste, sem entradas idênticas entre os conjuntos. Ajustamos o scaler de Time e Amount apenas no treino e aplicamos SMOTE ou undersampling somente nessa parte.

Nota de gravação: Mostrar as contagens no notebook e os três valores zero em feature_overlap_counts de reports/data_preparation.json. Diferenciar a base original da base limpa.

1:10 - 2:00 | MLP e hiperparâmetros

Mostrar

src/experiments.py: destacar as cinco configurações. Em src/train.py, destacar MLPClassifier e early_stopping.

Fala sugerida

Usamos MLPClassifier do scikit-learn, respeitando a orientação do professor de não usar TensorFlow, Keras ou PyTorch. A rede recebe 30 características e aprende a separar as duas classes. Testamos uma camada oculta com 32 neurônios e duas camadas com 64 e 32 ou 128 e 64 neurônios. Também variamos a taxa de aprendizado, a ativação ReLU ou tanh e o balanceamento. O treino usa Adam e parada antecipada. Depois ajustamos o limiar de classificação na validação, sem usar o teste para escolher a configuração.

Nota de gravação: São limites de 200 ou 300 iterações, não necessariamente o total executado. Não atribuir uma melhora exclusivamente ao balanceamento: o MLP-04 também muda a arquitetura.

2:00 - 3:25 | MLflow e escolha do modelo

Mostrar

MLflow: filtrar dataset_policy=exact_duplicates_removed_v1, comparar os cinco runs por métricas val_* e abrir os artefatos do MLP-03. Mostrar test_* apenas no vencedor.

Fala sugerida

Cada run registra a configuração, as métricas de validação e os artefatos. Precision mede quantos alertas são fraudes reais; recall mede quantas fraudes detectamos. O F1 resume o equilíbrio entre as duas medidas. Nossa regra prioriza recall de pelo menos 80% e escolhe o maior F1 entre os elegíveis, com ROC-AUC como desempate. Os cinco novos candidatos atingiram esse recall na validação. Venceu o MLP-03, com precision de 76%, recall de 80,28% e F1 de 78,08%. O limiar escolhido foi 0,83. A seleção usou a validação; o teste foi reservado para a avaliação final.

Nota de gravação: Não repetir a antiga justificativa de que MLP-02 tinha F1 maior e foi excluído pelo recall. Na nova execução, MLP-03 tem o maior F1 de validação. Não comparar diretamente os números antigos e novos como se usassem o mesmo conjunto de teste.

3:25 - 4:35 | API e predição

Mostrar

Mostrar a origem do modelo no run selecionado e o carregamento na API. No Swagger, colar os 30 campos, executar POST /predict e destacar HTTP 200, prediction, label e fraud_probability.

Fala sugerida

A API recebe as 30 características de uma transação. Carregamos o modelo selecionado a partir do MLflow, junto com o preprocessador e o limiar correspondentes ao mesmo run. Assim, a entrada recebe a mesma transformação usada no treinamento. Ao executar a requisição, a API retorna a classe, o rótulo e a probabilidade estimada de fraude. A classe é definida comparando essa estimativa com o limiar escolhido na validação.

Nota de gravação: A integração com MLflow foi implementada e testada. Mostrar o run_id retornado por /health e o carregamento em src/api.py. Explicar o resultado real mostrado na tela; não prometer antecipadamente fraude ou legítima. A saída é uma estimativa do modelo, não uma certeza.

4:35 - 5:00 | Resultado e conclusão

Mostrar

Relatório final de teste, com nome do modelo e métricas legíveis.

Fala sugerida

No teste, o MLP-03 teve precision de 66,67%, recall de 78,87% e F1 de 72,26%. Detectou 56 das 71 fraudes e gerou 28 falsos positivos. O recall ficou um pouco abaixo da referência de 80%. Mantivemos a escolha feita na validação, sem ajustar pelo teste, e disponibilizamos esse modelo pela API.

Nota de gravação: O teste contém 71 fraudes e 42.488 transações legítimas. Foram 56 verdadeiros positivos, 15 falsos negativos, 28 falsos positivos e 42.460 verdadeiros negativos. O ROC-AUC foi 0,9535. Não afirmar que todas as metas foram atingidas.
