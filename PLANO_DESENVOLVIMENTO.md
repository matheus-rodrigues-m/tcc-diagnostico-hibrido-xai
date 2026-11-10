# PLANO DE DESENVOLVIMENTO — TCC
**Desenvolvimento de uma Ferramenta Híbrida de Auxílio ao Diagnóstico Médico Utilizando Visão Computacional e Modelos Preditivos Explicáveis**
Discente: Matheus Rodrigues Silva Maia · Orientador: Prof. Matheus Wanderley Romão · UFSJ / Eng. Mecatrônica

> Protocolo: cada etapa = explicação + fonte → implementação atômica → verificações → **gate** ("Aprovado, podemos avançar para a etapa X").
> Legenda de status: ⬜ não iniciado · 🟨 em andamento / aguardando gate · ✅ aprovado

---

## Painel de status

| Fase | Descrição | Status |
|---|---|---|
| 0 | Ambiente, ingestão e integridade dos dados | 🟨 (0.1 ✅) |
| 1 | Dados tabulares: EDA, limpeza, pipelines anti-vazamento | ⬜ |
| 2 | Modelagem tabular: CV aninhada, comparação, calibração | ⬜ |
| 3 | XAI tabular: SHAP e LIME | ⬜ |
| 4 | Dados de visão: inspeção, splits por paciente, CLAHE, augmentation | ⬜ |
| 5 | Transfer learning, avaliação clínica, Grad-CAM, RSNA externo | ⬜ |
| 6 | Consolidação em `src/` + pytest | ⬜ |
| 7 | Integração: serviços + Streamlit | ⬜ |
| 8 | Consolidação científica para o texto do TCC | ⬜ |

---

## Checklist detalhado

### Fase 0 — Ambiente, ingestão e integridade
- [x] 0.1 Ambiente: uv + Python 3.12 + `pyproject.toml` + PyTorch CUDA (teste: `python scripts/check_env.py`) ✅
- [ ] 0.2 Estrutura de pastas, `git init`, `.gitignore`, `src/tcc_diag/{paths,seed}.py` (teste: `pytest tests/test_paths.py`) — 🟨 implementado e testado (16/16), aguardando gate
- [ ] 0.3 Ingestão Cleveland + sha256 (teste: 303×14; `?` só em ca(4)/thal(2); 164/139)
- [ ] 0.4 Download Kermany + integridade (5216/16/624; imagens abrem; ID de paciente; duplicatas pHash treino↔teste)
- [ ] 0.5 Download RSNA + integridade (26.684 DICOMs; labels coerentes; metadados)
- [ ] 0.6 Benchmark GPU (img/s e batch máx. por arquitetura) → decisão local × Kaggle

### Fase 1 — Dados tabulares
- [ ] 1.1 `01_tab_eda_clinica.ipynb` — EDA clínica, missingness informativa, outliers, plausibilidade
- [ ] 1.2 `02_tab_preprocessamento_pipeline.ipynb` — pipelines por família dentro do fold; teste de rótulos embaralhados (AUC≈0,5)

### Fase 2 — Modelagem tabular
- [ ] 2.1 `03_tab_modelagem_cv_aninhada.ipynb` — CV aninhada (3×10 externo / 5 interno / Optuna 50) + Dummy + LR
- [ ] 2.2 RF, SVM, XGBoost, LightGBM, CatBoost no mesmo framework
- [ ] 2.3 `04_tab_avaliacao_overfitting_calibracao.ipynb` — Friedman+Nemenyi, Nadeau–Bengio, regra 1-EP → campeão
- [ ] 2.4 Overfitting, calibração (Brier/reliability), limiar por OOF, MCC
- [ ] 2.5 Modelo final + artefato `models/heart/` (+ opcional: validação externa Hungarian)

### Fase 3 — XAI tabular
- [ ] 3.1 `05_tab_xai_shap.ipynb` — SHAP global (TreeSHAP), aditividade < 1e-6, estabilidade entre folds
- [ ] 3.2 SHAP para SVM/LR + concordância entre modelos
- [ ] 3.3 Explicações locais (waterfall) VP/VN/FP/FN
- [ ] 3.4 `06_tab_xai_lime_concordancia.ipynb` — LIME: kernel, 10 seeds, fidelidade, Kendall τ SHAP×LIME
- [ ] 3.5 Interpretação clínica × literatura, ALE, ressalvas de não causalidade

### Fase 4 — Dados de visão
- [ ] 4.1 `07_img_inspecao_datasets.ipynb` — inspeção Kermany/RSNA
- [ ] 4.2 Splits agrupados por paciente (teste oficial intocado); RSNA subconjunto balanceado
- [ ] 4.3 `08_img_preprocessamento_clahe_augmentation.ipynb` — HE/CLAHE, cache 256 px
- [ ] 4.4 Augmentation + DataLoader + pesos de classe

### Fase 5 — Transfer learning e avaliação clínica
- [ ] 5.1 `09_img_treino_ablacao.ipynb` — loop de treino; teste de overfit de 1 batch
- [ ] 5.2 Ablações (EfficientNet-B0): pré-processamento × estratégia de fine-tuning
- [ ] 5.3 `10_img_transfer_learning_principal.ipynb` — 3 arquiteturas × 3 seeds
- [ ] 5.4 `11_img_avaliacao_clinica_gradcam.ipynb` — limiar + temperature scaling; avaliação única no teste; IC bootstrap, DeLong, McNemar
- [ ] 5.5 Grad-CAM + sanity check de randomização
- [ ] 5.6 `12_img_validacao_externa_rsna.ipynb` — zero-shot, fine-tuning curto, Grad-CAM × caixas (IoU, pointing game)

### Fase 6 — Consolidação em `src/`
- [ ] 6.1 Refatoração para `src/tcc_diag/` (tabular, vision, inference)
- [ ] 6.2 Serialização + `models/manifest.json` (sha256) + model cards
- [ ] 6.3 Suíte pytest (integridade, vazamento, pipelines, aditividade SHAP, transforms, artefatos, paridade notebook↔src) + ruff

### Fase 7 — Integração e interface
- [ ] 7.1 Serviços `HeartRiskService` / `XRayService` + esquemas Pydantic + exceções
- [ ] 7.2 App Streamlit (Início, Raio-X, Risco Cardíaco, Sobre os Modelos)
- [ ] 7.3 Testes de entradas inválidas + e2e (`AppTest`) + latência

### Fase 8 — Consolidação científica
- [ ] 8.1 `13_consolidacao_resultados_tcc.ipynb` — tabelas finais (CSV/LaTeX) com IC
- [ ] 8.2 Figuras 300 dpi padronizadas
- [ ] 8.3 README de reprodutibilidade + limitações e ameaças à validade

---

## Metas de métricas

| Módulo | Métrica | Meta | Referência da literatura |
|---|---|---|---|
| Visão (teste oficial Kermany, 624) | ROC-AUC | ≥ 0,95 | Kermany 2018: 0,968 |
| | Sensibilidade | ≥ 0,95 | Kermany 2018: 93,2% |
| | Acurácia | ≥ 0,90 | Kermany 2018: 92,8% |
| | Especificidade | ≥ 0,80 (mínimo) | Kermany 2018: 90,1%; ResNet-50 puro (Sebastian): 65,4% |
| Visão (RSNA externo) | ROC-AUC | relatar (queda esperada) | Kundu 2021 (CV no RSNA): 0,87 |
| Tabular (CV aninhada) | ROC-AUC | ≥ 0,88 | — |
| | Acurácia | 0,83–0,87 | Detrano 1989: 77–81%; Ahmadian 2025 (CV): 86,2% |
| | MCC | ≥ 0,65 | — |
| XAI | Aditividade SHAP | erro < 1e-6 | Lundberg & Lee 2017 |
| | Estabilidade ranking SHAP entre folds | Spearman ρ ≥ 0,8 | — |
| | Estabilidade LIME (Jaccard top-5, 10 seeds) | ≥ 0,7 | — |
| | Grad-CAM randomização | mapa deve mudar | Adebayo et al. 2018 |
| App | Entradas inválidas tratadas | 100% sem travar | — |
| | Latência por raio-X + Grad-CAM | < 2 s GPU / < 5 s CPU | — |

> ⚠ Valores tabulares > 0,95 de acurácia indicam provável vazamento — investigar antes de comemorar.

---

## Notebooks previstos

| # | Notebook | Fase | Status |
|---|---|---|---|
| 00 | `00_ambiente_e_integridade.ipynb` | 0 | ⬜ |
| 01 | `01_tab_eda_clinica.ipynb` | 1 | ⬜ |
| 02 | `02_tab_preprocessamento_pipeline.ipynb` | 1 | ⬜ |
| 03 | `03_tab_modelagem_cv_aninhada.ipynb` | 2 | ⬜ |
| 04 | `04_tab_avaliacao_overfitting_calibracao.ipynb` | 2 | ⬜ |
| 05 | `05_tab_xai_shap.ipynb` | 3 | ⬜ |
| 06 | `06_tab_xai_lime_concordancia.ipynb` | 3 | ⬜ |
| 07 | `07_img_inspecao_datasets.ipynb` | 4 | ⬜ |
| 08 | `08_img_preprocessamento_clahe_augmentation.ipynb` | 4 | ⬜ |
| 09 | `09_img_treino_ablacao.ipynb` | 5 | ⬜ |
| 10 | `10_img_transfer_learning_principal.ipynb` | 5 | ⬜ |
| 11 | `11_img_avaliacao_clinica_gradcam.ipynb` | 5 | ⬜ |
| 12 | `12_img_validacao_externa_rsna.ipynb` | 5 | ⬜ |
| 13 | `13_consolidacao_resultados_tcc.ipynb` | 8 | ⬜ |

---

## Registro de decisões

| Data | Decisão | Justificativa |
|---|---|---|
| 2026-11-08 | RSNA = validação externa + fine-tuning curto + validação quantitativa do Grad-CAM | Generalização pediátrico→adulto; caixas permitem medir localização (IoU) |
| 2026-11-08 | Interface: Streamlit + camada de serviço em `src/` | Só Python; exibe imagens/SHAP/Grad-CAM; serviços testáveis |
| 2026-11-08 | Treino local (RTX 3050 4 GB) com Kaggle Notebooks como reserva | Velocidade similar à T4 grátis; dados em SSD; confirmar no benchmark 0.6 |
| 2026-11-08 | uv + Python 3.12 | Wheels maduras; mesma versão do Colab/Kaggle; `uv.lock` reprodutível |
| 2026-11-08 | Split agrupado por paciente; teste oficial Kermany intocado | Literatura com vazamento (Salehi 2021, Kermany 2018 p13, Sebastian 2025) |
| 2026-11-08 | Só `processed.cleveland.data` (303); nunca o heart.csv do Kaggle (1025) | ~723 duplicatas inflam resultados (Upadhyayula 2025) |
| 2026-11-08 | CV aninhada repetida (3×10 / 5) + LR e Dummy como baselines | Kuhn & Johnson p.70, 78, 500; Detrano 1989 |
| 2026-11-08 | Sem flip horizontal/vertical na augmentation | Inverteria lateralidade cardíaca; Shorten 2019 p7–9 |
| 2026-11-10 | Commits e pushes feitos pelo discente (assistente apenas fornece os comandos) | Controle total do histórico do repositório |
| 2026-11-10 | `explore/`, `data/` e pesos de modelos fora do git | PDFs com direitos autorais; dados regeneráveis por script; arquivos grandes |

## Registro de gates aprovados

| Etapa | Data | Observações |
|---|---|---|
| Plano mestre | 2026-11-09 | Aprovado pelo discente |
| 0.1 Ambiente | 2026-11-10 | 28 libs OK; RTX 3050 4 GB; teste AMP OK |

## Pendências do discente
- [x] Gerar token da API do Kaggle (necessário na etapa 0.4) — em `~/.kaggle/access_token`, testado
- [x] Aceitar as regras da competição RSNA no Kaggle (necessário na etapa 0.5) — confirmado
- [ ] (Opcional) Substituir `explore/7.../3290605.3300334.pdf` (Swire) por Amershi et al. 2019 — *Guidelines for Human-AI Interaction*
