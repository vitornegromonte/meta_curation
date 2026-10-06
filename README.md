# projeto_kunumi

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Guia de execução

### 1. Ambiente

Só CPU basta. Com venv própria:

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy scikit-learn pytest
```

Ou deixe o script criar tudo: `./scripts/compare_with_nara.sh --setup-env`
(cria `.venv-fulltest/` e usa automaticamente).

### 2. Teste completo em um dataset (recomendado)

```bash
# confronto DataRater vs Data-IQ do NARA no parkinson (fase 2)
./scripts/compare_with_nara.sh --dataset parkinson --meta-steps 600

# outro dataset + inclui fase 1 (curadoria dos dados reais)
./scripts/compare_with_nara.sh --dataset cholesterol --phase1 --meta-steps 300

# versão rápida (~2 min) para validar a fiação
./scripts/compare_with_nara.sh --dataset parkinson --meta-steps 2 --n-synth 800 --out-dir /tmp/quick
```

Flags: `--dataset` (parkinson|cholesterol|diabetes|fat|plasma|urinary),
`--meta-steps`, `--n-synth`, `--junk-frac`, `--out-dir`, `--phase1`,
`--setup-env`, `--python`. Datasets com <500 linhas emitem aviso
(meta-aprendizado fica ruidoso). Saída: `<out-dir>/phase2/run.pt`,
`phase2.log`, `summary.txt` (tabela comparativa).

### 3. Comandos diretos (módulos)

```bash
pytest tests -q                          # testes unitários
python -m meta_curation.mixflow          # self-test do modo misto
python -m meta_curation.implicit          # paridade explícito-vs-implícito
python -m meta_curation.nara_adapter --dataset parkinson --noise-frac 0.3 --keep 0.7
python -m meta_curation.nara_adapter --implicit --noise-frac 0.3 --keep 0.7   # via iMAML
python -m meta_curation.nara_phase2 --dataset parkinson --meta-steps 600
```

`--implicit` troca o meta-gradiente desenrolado pelo implícito (módulo
`implicit.py`, branch `feat/implicit-metagrad`).

### 4. O que cada artefato contém

`run.pt`: `scores` do rater, máscara de junk/ruído (só diagnóstico),
índices kept por braço, `test_mse` por braço e `args`. `summary.txt` é a
mesma tabela em texto.

## Smoke test

Requer `torch>=2.1` (CPU basta). Cada comando roda em ~1 min e
verifica o próprio sucesso:

```bash
pytest tests -q                          # testes unitários (otimizadores, filtragem)
python -m meta_curation.mixflow          # meta-gradiente em modo misto == modo reverso
python -m meta_curation.nara_adapter --meta-steps 2 --out-dir /tmp/smoke_adapter
python -m meta_curation.nara_phase2 --n-synth 800 --meta-steps 2 --out-dir /tmp/smoke_p2
```

Execuções completas (resultados de referência em `reports/`): remova as
flags de smoke, ex.
`python -m meta_curation.nara_phase2 --meta-steps 600` reproduz o
confronto DataRater-vs-DataIQ no pool sintético de parkinson
(`reports/nara_phase2/run.pt`).

## Resultados de referência

Parkinson, MSE de teste do modelo final (unidades de y padronizado, 3 seeds
de avaliação). Execuções completas:

Fase 1 — 30% de rótulos corrompidos, 600 meta-passos, keep 0.7
(`reports/nara_parkinson_noisy600/run.pt`):

| full | curated | random | noisy kept |
|---|---|---|---|
| 0.172 | 0.170 | 0.251 | 30.6% → 20.4% |

Fase 1 — rótulos limpos, 300 meta-passos, keep 0.75
(`reports/nara_parkinson_clean300/run.pt`): full 0.014, curated 0.056,
random 0.015. Podar exemplos difíceis-mas-limpos prejudica — curadoria é para
pools sujos, não para os limpos.

Fase 2 — pool sintético (8000 linhas, 30% de lixo), 600 meta-passos
(`reports/nara_phase2/run.pt`), corr(score, junk) = −0.77:

| braço | kept | junk-in-kept | test MSE |
|---|---|---|---|
| dataiq-lr (caminho exato do nara) | 8000 | 0.300 | 0.060 |
| dataiq-mlp | 8000 | 0.300 | 0.062 |
| datarater (mesmo orçamento) | 8000 | 0.300 | 0.058 |
| **datarater70** (keep 0.7) | 5600 | **0.063** | **0.037** |
| random70 (keep 0.7) | 5600 | 0.299 | 0.066 |

O Data-IQ como codificado não encontra nenhuma linha hard aqui (seus cortes
0.25/0.75 são limiares da era de classificação aplicados a valores-alvo
brutos), então mantém tudo; com orçamentos de keep iguais, o DataRater remove
~80% do lixo e quase reduz à metade o MSE de teste vs aleatório.

As execuções de smoke imprimem o mesmo formato de saída em escala mínima
(2 meta-passos); os valores exatos têm semente fixa, mas o que importa ali é
exit-0 + `run.pt`.

## Organização do Projeto

```
├── Makefile           <- Makefile com comandos de conveniência como `make data` ou `make train`
├── README.md          <- O README principal para desenvolvedores usando este projeto.
├── scripts
│   └── compare_with_nara.sh  <- Teste completo: DataRater vs Data-IQ do NARA
│
├── tests                <- Testes unitários (pytest)
│
├── data
│   ├── external       <- Dados de fontes externas.
│   ├── interim        <- Dados intermediários que já foram transformados.
│   ├── processed      <- Os conjuntos de dados finais e canônicos para modelagem.
│   └── raw            <- O despejo original e imutável dos dados.
│
├── docs               <- Um projeto mkdocs padrão; ver www.mkdocs.org para detalhes
│
├── models             <- Modelos treinados e serializados, predições ou resumos de modelos
│
├── notebooks          <- Jupyter notebooks. A convenção de nome é um número (para ordenação),
│                         as iniciais do criador e uma descrição curta separada por `-`, ex.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Arquivo de configuração do projeto com metadados do pacote
│                         meta_curation e configuração de ferramentas como black
│
├── references         <- Dicionários de dados, manuais e todo outro material explicativo.
│
├── reports            <- Análises geradas como HTML, PDF, LaTeX, etc.
│   └── figures        <- Gráficos e figuras gerados para uso nos relatórios
│
├── requirements.txt   <- O arquivo de requirements para reproduzir o ambiente de análise, ex.
│                         gerado com `pip freeze > requirements.txt`
│
├── setup.cfg          <- Arquivo de configuração para flake8
│
└── meta_curation   <- Código-fonte para uso neste projeto.
    │
    ├── __init__.py             <- Re-exporta a API pública do DataRater
    │
    ├── types.py                <- Aliases compartilhados (ParamDict, Batch, ...)
    ├── config.py               <- DataRaterConfig (§3 do single-file.py)
    ├── optim.py                <- Otimizadores internos diferenciáveis (§1)
    ├── meta_optim.py           <- Meta-otimizador MetaAdam (§2)
    ├── trainer.py              <- DataRaterTrainer, Algoritmo 1 (§4)
    ├── filtering.py            <- Utilitários de filtragem Top-K / CDF (§5)
    ├── demo.py                 <- Demo toy (§7, `python -m meta_curation.demo`)
    ├── mixflow.py              <- Meta-gradientes em modo misto (fwd-over-rev)
    ├── implicit.py             <- Meta-gradientes implícitos estilo iMAML
    ├── nara_adapter.py         <- Curadoria DataRater para dados tabulares do artigo da Maynara (Fase 1)
    ├── nara_phase2.py          <- Confronto DataRater-vs-DataIQ em pools sintéticos
    ├── single-file.py          <- Implementação de referência em arquivo único (mantida como referência)
    │
    └── modeling
        ├── __init__.py
        └── models.py           <- Redes DataRater de exemplo (§6)
```

--------
