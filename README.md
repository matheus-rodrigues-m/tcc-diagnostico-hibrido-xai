# tcc-diagnostico-hibrido-xai

Ferramenta híbrida de apoio ao diagnóstico médico (TCC Eng. Mecatrônica – UFSJ): detecção de pneumonia em raio-X com Transfer Learning (ResNet-50, DenseNet-121, EfficientNet-B0) e predição explicável de risco cardíaco (RF, SVM, XGBoost, LightGBM, CatBoost) com SHAP, LIME e Grad-CAM.

**Discente:** Matheus Rodrigues Silva Maia · **Orientador:** Prof. Matheus Wanderley Romão

> ⚠ Projeto acadêmico. Ferramenta de **apoio à decisão**, não substitui avaliação médica.

## Andamento
Ver [`PLANO_DESENVOLVIMENTO.md`](PLANO_DESENVOLVIMENTO.md).

## Ambiente (resumo)
```bash
uv sync                              # cria .venv com Python 3.12 e dependências travadas (uv.lock)
uv run python scripts/check_env.py   # verifica bibliotecas e GPU
uv run pytest                        # testes
```
Instruções completas de reprodutibilidade (download dos dados, ordem dos notebooks, app) serão adicionadas na Fase 8.
