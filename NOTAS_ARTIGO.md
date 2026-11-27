# Notas para o Artigo Científico

> Arquivo vivo, atualizado ao final de cada etapa. Reúne o que deve constar no artigo e que
> não fica evidente só pelo código: números exatos, justificativas, versões, armadilhas da
> literatura e ameaças à validade. **Não é o texto do artigo**: é a matéria-prima organizada.

---

## 1. Posicionamento e contribuição

### 1.1 Mensagem central (rascunho)
Ferramentas de apoio ao diagnóstico só são úteis se (a) o desempenho reportado for **honesto**
e (b) as explicações forem **verificáveis**. O artigo mostra um sistema com dois módulos
(pneumonia em raio-X; risco cardíaco tabular) avaliados com protocolo livre de vazamento, e com
explicações (SHAP, LIME, Grad-CAM) **validadas quantitativamente**, e não só exibidas.

### 1.2 Contribuições candidatas (confirmar com os resultados)
1. **Protocolo de avaliação rigoroso** em contraste com a literatura revisada:
   - split por paciente;
   - teste oficial intocado;
   - CV aninhada no tabular;
   - IC 95% e testes estatísticos.
2. **Auditoria metodológica da literatura de base**: vazamento e seleção no teste identificados
   em artigos muito citados (ver §4), com página.
3. **Validação das explicações**:
   - aditividade e estabilidade do SHAP;
   - estabilidade e fidelidade do LIME;
   - sanity check do Grad-CAM;
   - localização do Grad-CAM medida contra as caixas do RSNA.
4. **Validação externa / mudança de domínio**: Kermany (pediátrico) → RSNA (adulto).
5. **Ferramenta integrada** com explicações, incerteza e avisos de uso (alinhada a Abgrall 2024, Tab. 1).

### 1.3 Ressalva de escopo (declarar explicitamente)
Não há dataset com raio-X e dados clínicos do mesmo paciente. A ferramenta é uma
**plataforma unificada com dois módulos independentes**, sem fusão multimodal. Não chamar de
"modelo multimodal".

### 1.4 Veículos possíveis (avaliar com o orientador)
- **SBCAS**: Simpósio Brasileiro de Computação Aplicada à Saúde (SBC).
- **CBEB**: Congresso Brasileiro de Engenharia Biomédica.
- **Research on Biomedical Engineering** (Springer; antiga Revista Brasileira de Eng. Biomédica).

### 1.5 Diretrizes de relato a seguir
- **TRIPOD+AI** (Collins et al., BMJ 2024): modelos de predição clínica, para o módulo tabular.
- **CLAIM** (Checklist for AI in Medical Imaging, atualização 2024): módulo de visão.
- Montar a tabela de conformidade com o checklist como material suplementar.

---

## 2. Métodos: detalhes de reprodutibilidade

### 2.1 Ambiente computacional (Etapa 0.1)
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU, **4 GB VRAM**, compute capability 8.6; Intel Core i5-12500H; 24 GB RAM; Windows 11 Pro.
- **Software:** Python 3.12.16, gerenciado por **uv 0.13.7**, com dependências travadas em `uv.lock`.
- **Versões-chave:**

  | Pacote | Versão |
  |---|---|
  | PyTorch | 2.11.0 (CUDA 12.8; cuDNN 9.19) |
  | torchvision | 0.26.0 |
  | scikit-learn | 1.9.1 |
  | XGBoost | 3.4.1 |
  | LightGBM | 4.7.0 |
  | CatBoost | 1.2.10 |
  | SHAP | 0.54.1 |
  | Optuna | 5.0.0 |
  | OpenCV | 5.0.0 |
  | pydicom | 3.0.2 |
  | Streamlit | 1.67.0 |

- **Relevância:** a VRAM de 4 GB **restringe batch e resolução**. Relatar batch size e uso de precisão mista (AMP) e citar como limitação e escolha de engenharia (hardware acessível).

### 2.2 Reprodutibilidade (Etapa 0.2)
- **Sementes fixas** (`set_seed`): Python, NumPy, PyTorch CPU/GPU; cuDNN determinístico; `CUBLAS_WORKSPACE_CONFIG`. Seed padrão 42; relatar as seeds de cada experimento.
- **Testado:** mesma seed → mesma sequência, inclusive na GPU.
- **Ressalva:** operações de GPU podem ter diferenças residuais (~1e-6) mesmo com determinismo.
- **Caminhos centralizados** (`paths.py`): o mesmo código roda local e no Kaggle (`TCC_DATA_DIR`).
- **Versionamento:**
  - Código em git (GitHub).
  - Dados **fora** do git, regeneráveis por script e verificados por SHA-256.
  - Pesos de modelos fora do git, com manifesto de hash (Fase 6).
- **Base conceitual:** Sculley et al. (2015): dívidas de reprodutibilidade (p8), de configuração (p6) e *glue code* (p5).

### 2.3 Dados: UCI Heart Disease, Cleveland (Etapa 0.3)
- **Arquivo:** `processed.cleveland.data` (UCI id 45), SHA-256 `a74b7efa387bc9d1…` (completo em `configs/data_manifest.json`).
- **Composição:**
  - 303 pacientes, 13 preditores + alvo `num` (0–4).
  - Distribuição 164/55/36/35/13, **idêntica** à seção 10 do `heart-disease.names`.
- **Alvo binário:** presença (`num` 1–4) × ausência (0), convenção da própria documentação (seção 4). Resultado: **139 positivos / 164 negativos (prevalência 45,9%)**, classes quase balanceadas, sem necessidade de reamostragem.
- **Ausentes:** 6 células em 6 pacientes, `ca` (4) e `thal` (2). Dos 6, 5 são negativos e 1 positivo. O marcador no arquivo processado é `?`; na documentação é `-9.0`, que vale para os arquivos não processados. Verificado que não há `-9` mascarado.
- **0 duplicatas exatas.**
- **Faixas observadas:**

  | Variável | Mínimo | Mediana | Máximo |
  |---|---|---|---|
  | Idade (anos) | 29 | 56 | 77 |
  | PA de repouso (mm Hg) | 94 | 130 | 200 |
  | Colesterol (mg/dl) | 126 | 241 | 564 |
  | FC máxima (bpm) | 71 | 153 | 202 |
  | oldpeak (mm) | 0 | 0,8 | 6,2 |

- **Arquivo descartado:** o `cleveland.data` bruto, corrompido segundo o aviso `WARNING` da própria UCI. Mencionar em uma frase.
- **Copiados para validação externa opcional:** Hungarian (294), Switzerland (123), VA (200), no espírito de Detrano et al. (1989).

### 2.4 Dados: Chest X-Ray, Kermany/Guangzhou (Etapa 0.4)
- **Fonte e arquivo:** Kaggle `paultimothymooney/chest-xray-pneumonia` (versão de 2018-03-24), .zip de 2,46 GB (= 2,29 GiB). SHA-256 do .zip e hash agregado das 5.856 imagens em `configs/data_manifest.json`.
- **Armadilhas do .zip** (relatar como nota de reprodutibilidade):
  - O arquivo contém **duas cópias** do dataset (`chest_xray/` e `chest_xray/chest_xray/`). Verificamos que são idênticas (nomes + CRC32), e só uma foi extraída.
  - Também traz metadados `__MACOSX`, descartados.
  - Quem usar `glob` recursivo sem cuidado **dobra** o dataset.
- **Contagens** (idênticas às esperadas):

  | Split | Normal | Pneumonia | Total |
  |---|---|---|---|
  | Treino | 1.341 | 3.875 | 5.216 |
  | Validação | 8 | 8 | 16 |
  | Teste | 234 | 390 | 624 |

  A diferença para Kermany 2018 (p7: treino 5.232 = 1.349 N + 3.883 P) são as 16 imagens que o Kaggle moveu para "val".
- **Proporção:** ~3:1 pneumonia:normal no treino; 1,67:1 no teste.
- **Subtipos de pneumonia:** bacteriana 2.780, viral 1.493 (total 4.273 = 3.875 + 8 + 390) (só informativo; a tarefa é binária).
- **Qualidade:**
  - 0 imagens ilegíveis.
  - 5.573 em tons de cinza (L) e **283 RGB**.
  - Larguras de 384 a 2.916 px (mediana 1.281); alturas de 127 a 2.713 px (mediana 888); razão L/A de 0,84 a 3,38 (mediana 1,42).
  - A resolução heterogênea justifica o redimensionamento padronizado.
- **ID do paciente pelo nome do arquivo:**
  - Padrões: `person{ID}_{bacteria|virus}_{n}[_k]`, `[NORMAL2-]IM-{ID}-{seq}[-{sub}]`.
  - 100% das imagens com ID extraído.
  - Pacientes únicos: treino 1.211 N / 1.635 P; teste 225 N / 202 P; até ~2,4 imagens por paciente (pneumonia, treino).
  - **Implicação:** o split por imagem coloca o mesmo paciente em treino e validação.
- **Conflito com o artigo original:** Kermany 2018 (p7) fala em teste com "624 patients", mas pelos nomes de arquivo o teste tem 427 IDs. Também diz "5,856 patients" para o total, o que é inconsistente. Os números de pacientes do artigo original não são confiáveis; reportar a contagem pelos IDs, com essa ressalva.
- **Duplicatas:**
  - 30 grupos de duplicatas **exatas** (SHA-256), 62 imagens, **todas dentro do mesmo split** (24 no treino, 6 no teste). Nenhuma cruza treino↔teste.
  - 1 par de duplicatas exatas com **IDs diferentes** (`NORMAL2-IM-0095`/`0096`, teste), o que indica erro de rotulagem de paciente. Visualmente, essa radiografia **parece de adulto** num dataset declarado pediátrico (1–5 anos): observação qualitativa, a mencionar com cautela.
  - O teste contém 12 imagens em grupos duplicados, o que dá peso duplo a alguns casos nas métricas. Avaliar o impacto na Fase 5 (métricas com e sem deduplicação).
- **Quase-duplicatas (pHash 64 bits):**
  - Pares treino↔teste: 0 com dist = 0, 1 com ≤ 2, 26 com ≤ 4, 284 com ≤ 6.
  - **A inspeção visual dos pares mais próximos mostrou crianças diferentes**, inclusive de classes diferentes.
  - **Conclusão metodológica:** em radiografias de tórax, o pHash com limiar > 0 não serve para detectar duplicatas, porque as imagens são intrinsecamente parecidas. Só dist = 0 / SHA-256 são confiáveis. Útil como nota metodológica.
- **IDs coincidentes treino↔teste:**
  - **Normais:** faixas de ID disjuntas (teste 1–381 × treino 115–1.423, por prefixo), o que é consistente com um bloco de teste separado.
  - **Pneumonia:** 170 IDs coincidem, cobrindo **320 das 390** imagens de pneumonia do teste. Porém a similaridade visual entre imagens com o mesmo ID **não difere** da de pares sorteados (mediana de 22 bits em ambos; Mann-Whitney p = 0,69).
  - **Conclusão:** numeração independente por pasta é a explicação mais provável. Declarar como ameaça (T7), com a evidência.
- **Potencial atalho (*shortcut*):** **todas as 283 imagens RGB estão em treino/PNEUMONIA**. Sem conversão para tons de cinza, o modelo poderia aprender "cor ⇒ pneumonia". Mitigação: conversão obrigatória para cinza (Fase 4) e teste automatizado. Bom exemplo para a discussão sobre *shortcut learning* (Molnar; husky × lobo de Ribeiro 2016).

---

## 3. Ameaças à validade (lista corrente)

| # | Ameaça | Mitigação / como relatar |
|---|---|---|
| T1 | Cleveland pequeno (n = 303): estimativas com alta variância | CV aninhada repetida; IC; relatar média ± dp |
| T2 | Dados de 1988, população de um único centro (Cleveland Clinic) | Validação externa opcional (Hungarian/VA); discutir generalização temporal e geográfica |
| T3 | Ausência informativa em `ca`/`thal` com só 6 casos | Teste com baixo poder; reportar como inconclusivo se for o caso |
| T4 | Hardware limitado (4 GB VRAM) | Relatar batch/AMP; resultados podem ser melhorados com mais memória |
| T5 | Sem fusão multimodal (pacientes distintos) | Declarar como limitação de dados públicos |
| T6 | ID de paciente inferido do nome do arquivo (sem metadados oficiais) | Relatar a regra; agrupar também por duplicatas exatas (união de grupos) na Fase 4 |
| T7 | 170 IDs de pneumonia coincidem entre treino e teste (320/390 imagens de teste) | Evidência de numeração independente (pHash, p = 0,69); análise de sensibilidade opcional na Fase 5 |
| T8 | Duplicatas exatas dentro do teste (12 imagens) | Reportar métricas também no teste deduplicado |
| T9 | Atalho RGB ↔ pneumonia no treino | Conversão para cinza obrigatória + teste; Grad-CAM para verificar atalhos (marcadores "R", textos) |
| T10 | Kermany é pediátrico e de um único centro (Guangzhou) | Validação externa no RSNA (adultos, NIH) |

---

## 4. Auditoria da literatura (material para Introdução/Discussão)

### 4.1 Pneumonia (Kermany e afins)
| Trabalho | Achado metodológico | Página |
|---|---|---|
| Kermany et al. 2018 (Cell) | Teste usado para selecionar o modelo ("best performing model was kept") | p13 |
| Salehi et al. 2021 (BJR) | SMOTE/augmentation **antes** do split 80/20; os autores admitem o mesmo paciente em treino e teste | p3, p6 |
| Sebastian & Ankayarkanni 2025 | Re-split 70/15/15 **por imagem**, sem agrupar por paciente; ResNet-50 sem tratamento de desbalanceamento tem especificidade de 65,4% | p6–7, Tab. 6 |
| Kundu et al. 2021 (PLOS ONE) | CV 5-fold por imagem; sem conjunto de validação separado | p13–15 |
| Buriboev et al. 2025 (Sensors) | Splits inconsistentes (4173/624/1066 × 70/20/10); CL do CLAHE em escala [0,1] ≠ `clipLimit` do OpenCV | p4, p20; p5–9 |

### 4.2 Doença cardíaca
| Trabalho | Achado | Observação |
|---|---|---|
| Upadhyayula & Pothugunta 2025 (medRxiv) | Usa o heart.csv do Kaggle (1025 linhas, ~723 duplicatas) | Acurácia de 0,97–0,98 inflada |
| Alsabhan & Alfadhly 2025 (Sci Rep) | SMOTE aparentemente antes do split; único split 80/20 | Inferência pelos denominadores k/204 |
| Nissa et al. 2024 (Computation) | Não usa o Cleveland (Kaggle, 8763 linhas) | Não comparável |
| Ahmadian et al. 2025 (Algorithms) | Hold-out de 60 linhas: 93,3%; **CV 10-fold: 86,2%** | Usar o número de CV como referência |

**Faixa honesta no Cleveland-303:** ~77–87% de acurácia (Detrano 1989: 77–81% externo; Ahmadian 2025: 86,2% em CV).

### 4.3 Referências a incluir que não estão na pasta `explore/`
- **Visão / XAI:** Selvaraju 2017 (Grad-CAM); Adebayo 2018 (sanity checks); Guo 2017 (temperature scaling).
- **Estatística e validação:** DeLong 1988; Nadeau & Bengio 2003; Demšar 2006; Varma & Simon 2006; Cawley & Talbot 2010; Yeo & Johnson 2000.
- **Diretrizes e interação humano-IA:** Collins 2024 (TRIPOD+AI); CLAIM 2024; Amershi 2019 (Human-AI guidelines).

---

## 5. Figuras e tabelas planejadas
- [ ] Fluxograma do sistema (dois módulos → serviço → interface)
- [ ] Diagrama do protocolo de avaliação (split por paciente; CV aninhada)
- [ ] Tabela 1: características dos datasets (n, prevalência, população, ausentes)
- [ ] Tabela comparativa com a literatura, com coluna "protocolo de validação"
- [ ] ROC/PR com IC; matrizes de confusão
- [ ] SHAP beeswarm + waterfall; estabilidade do LIME
- [ ] Grad-CAM: exemplos VP/FP/FN + IoU com as caixas do RSNA
- [ ] Captura da interface

---

## 6. Diário por etapa
| Etapa | Data | Nota para o artigo |
|---|---|---|
| 0.1 | 2026-11-10 | Ambiente e versões (§2.1) |
| 0.2 | 2026-11-10 | Sementes, caminhos, versionamento (§2.2) |
| 0.3 | 2026-11-25 | Cleveland verificado contra a documentação oficial (§2.3) |
| 0.4 | 2026-11-25 | Kermany: cópia duplicada no .zip, IDs de paciente, duplicatas, atalho RGB (§2.4; T6–T10) |
