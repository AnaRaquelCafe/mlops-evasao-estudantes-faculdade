# Student Success MLOps — Predição de Evasão Universitária

> **Objetivo:** classificar a situação acadêmica com base nos rótulos disponíveis na base (por exemplo, `Dropout`, `Enrolled` e `Graduate`) e apoiar a priorização de ações de acolhimento. A previsão é um sinal para análise humana, nunca uma decisão automática sobre estudantes.

## Arquitetura

```text
Arquivo XLSX/CSV no Volume do Unity Catalog
        |
        v
01_ingest_and_validate.py  -> validação + tabela Delta Bronze
        |
        v
02_train_evaluate_register.py -> preprocessing + treino + métricas + MLflow
        |
        v
Unity Catalog / Model Registry -> versão rastreável
        |
        +--> 03_batch_inference.py -> scores em tabela Delta
        |
        +--> Model Serving (opcional) -> endpoint REST
```

## Estrutura

```text
evasao-faculdade-mlops/
├── notebooks/
│   ├── 01_ingest_and_validate.py
│   ├── 02_train_evaluate_register.py
│   └── 03_batch_inference.py
├── src/evasao/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── train.py
├── tests/test_data.py
├── resources/catalog_setup.sql
├── requirements.txt
└── README.md
```

## Executar na Databricks Free Edition

A Free Edition usa compute serverless e possui limites próprios. Este fluxo evita depender de Docker, GPU ou cluster customizado. Recursos disponíveis podem variar por workspace.

1. Crie ou selecione um catálogo e schema no Unity Catalog.
2. Crie um Volume para a base e envie `StudentsPrepared.xlsx`.
3. Importe os três arquivos da pasta `notebooks/` para o workspace como notebooks Python.
4. Edite `CATALOG`, `SCHEMA`, `VOLUME`, `SOURCE_FILE` e `TARGET_COLUMN` no início dos notebooks.
5. Execute os notebooks em ordem: `01` → `02` → `03`.
6. Se Model Serving estiver disponível, publique a versão registrada. Caso contrário, execute inferência batch.

### Criar schema e volume

Execute em um SQL Editor com as permissões necessárias:

```sql
CREATE SCHEMA IF NOT EXISTS main.evasao_mlops;
CREATE VOLUME IF NOT EXISTS main.evasao_mlops.raw_files;
```

Envie a base para:

`/Volumes/main/evasao_mlops/raw_files/StudentsPrepared.xlsx`

Se seu catálogo não for `main`, troque o valor em todos os notebooks.

## Importante: confirme o target

O pipeline tenta identificar colunas chamadas `Target`, `target`, `Status`, `situacao` ou equivalentes. Confirme o nome exato no notebook 01 e preencha `TARGET_COLUMN` nos notebooks 02 e 03 se necessário.

Para prever risco de evasão **antes** do desfecho, remova variáveis que só se tornam conhecidas depois do período de intervenção. Caso contrário, haverá vazamento de dados (*data leakage*) e métricas artificialmente altas.

## Métricas e rastreabilidade

O notebook 02 compara Logistic Regression e Random Forest. A seleção inicial usa macro-F1 para dar peso semelhante às classes, mas a escolha final deve considerar a matriz de confusão e o custo dos erros no contexto acadêmico. Experimentos e artefatos são registrados no MLflow; o registro no Unity Catalog requer permissões adequadas.

## Inferência

O notebook 03 lê uma tabela Delta, carrega um modelo registrado via `MODEL_URI` e grava previsões em `student_dropout_predictions`. Configure `MODEL_URI`, por exemplo:

`models:/main.evasao_mlops.student_dropout_classifier/1`

A inferência só funciona se as colunas de entrada corresponderem às utilizadas no treinamento. Em um cenário real, crie uma tabela separada de alunos elegíveis para previsão e não use a própria base histórica com o target como se fosse produção.

## Uso responsável

Use os resultados para orientar apoio, contato e acompanhamento, não para negar oportunidades ou penalizar estudantes. Avalie vieses, desempenho por grupo quando apropriado, privacidade e explicabilidade antes de qualquer uso real.

## Referências

- [Case original](https://github.com/AnaRaquelCafe/PredizendoEvasaoFaculdade)
- [Databricks Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition)
- [Model Serving — modelos customizados](https://docs.databricks.com/aws/en/machine-learning/model-serving/custom-models)
