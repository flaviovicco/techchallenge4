# Tech Challenge 4 - 5IADT
Tech Challenge - Fase 4: Exemplo de Analise de Video em Python para detectar faces, emoções e atividades
da Pos Tech em IA para Devs da FIAP, 2025.

Flavio Luiz Vicco - RM 361664

https://youtu.be/

# 🎬 Análise de Vídeo com IA  
Detecção de Faces, Emoções e Ações em PT-BR

## 🎯 1. Visão Geral

Este projeto implementa um pipeline completo de **análise de vídeo com IA**, focado em:

- Detectar **rostos** em cada frame do vídeo.
- Classificar **emoções faciais** (via DeepFace) e traduzir para **português**.
- Gerar **legendas descritivas da cena** (via BLIP) e traduzir EN → PT-BR.
- Atribuir um **identificador único por pessoa** (`pessoa1`, `pessoa2`, …) ao longo de todo o vídeo.
- Produzir:
  - Um **vídeo anotado** (rostos + emoção + pessoa + legenda).
  - Um **CSV detalhado** frame a frame.
  - Um **CSV de resumo**, consolidando emoções e ações **por pessoa**, pronto para análise de negócios.

---

## 🧠 2. Funcionalidades Principais

- **Detecção de Faces (MediaPipe Tasks)**
  - Uso do modelo BlazeFace (`.tflite`) via MediaPipe Tasks.
  - Download automático do modelo se não existir localmente.
  - Retorna bounding boxes em coordenadas de pixels + score de confiança.

- **Análise de Emoções (DeepFace)**
  - Ação: `emotion`.
  - Normalização do retorno:
    - Tratar tanto dict quanto lista de resultados.
    - Trabalhar com `dominant_emotion` + dicionário de scores.
  - Fallback: se `dominant_emotion` vier vazio, escolhe a emoção com maior score.

- **Tradução de Emoções para PT-BR**
  - Mapeamento fixo:
    - `happy → Alegre`, `sad → Triste`, `neutral → Neutro`, etc.
  - Garantia de que o vídeo e os CSVs estejam **100% em português** nas emoções.

- **Descrição de Ações (BLIP)**
  - Geração de legenda em inglês com o modelo `Salesforce/blip-image-captioning-base`.
  - Tradução para PT-BR usando modelo T5 (`unicamp-dl/translation-en-pt-t5`).
  - Limpeza de acentuação para desenhar texto com mais segurança em OpenCV.

- **Identificação Única de Pessoas**
  - Rastreamento simples por proximidade do centro da bounding box.
  - Mesma pessoa ao longo do vídeo recebe sempre o mesmo nome:
    - `pessoa1`, `pessoa2`, `pessoa3`, ...
  - Útil para:
    - Contagem de pessoas.
    - Análise de emoção por indivíduo.
    - Narrativas do tipo “persona”.

- **Geração de Saídas de Negócio**
  - **Vídeo anotado** com:
    - Retângulo no rosto.
    - Texto: `pessoaX - Emoção`.
    - Legenda da cena no rodapé em PT-BR.
  - **CSV detalhado** (`resultado.csv`) com:
    - Frame, pessoa, emoção, score, ação.
  - **CSV resumo** (`resultado_resumo.csv`) consolidando:
    - Uma emoção representativa por pessoa.
    - Uma ação representativa por pessoa.
    - Totais por emoção e por ação.

---

## 🧩 3. Arquitetura do Pipeline

### 3.1. Etapas do Fluxo

1. **Leitura do vídeo**
   - `cv2.VideoCapture(input_path)`

2. **Inicialização dos modelos**
   - `init_mediapipe_face_detector(conf)`  
   - `init_blip_colab(device="cpu")`  
   - `init_translator_en_pt(device="cpu")`

3. **Loop de frames**
   - Leitura de cada frame.
   - Conversão BGR → RGB.
   - Detecção de faces (MediaPipe Tasks).
   - Rastreamento de pessoas com base na proximidade das bbox.
   - Para cada rosto:
     - Recorte da face.
     - Análise de emoção (DeepFace).
     - Tradução da emoção para PT-BR.
     - Desenho da label: `pessoaX - Emoção`.
   - Geração da legenda da cena (BLIP + tradução EN→PT).
   - Desenho da legenda no rodapé do frame.
   - Escrita do frame anotado no vídeo de saída.

4. 🔁 **Geração de CSV detalhado**
   - Cada linha = 1 pessoa em 1 frame:
     - `frame`, `face_id`, `person_label`, `emotion`, `score`, `caption`.

5. **Geração de CSV de resumo**
   - Agrupa linhas por `person_label`.
   - Para cada pessoa:
     - Emoção representativa = emoção mais frequente.
     - Ação representativa = legenda mais frequente.
   - Conta quantas pessoas têm cada emoção/ação.

---

## 📊 4. Estrutura dos Arquivos

### 4.1. Vídeo de Saída

- **Conteúdo:**
  - Retângulos verdes em volta das faces.
  - Texto no topo da bbox:
    - `pessoa1 - Alegre`, `pessoa2 - Triste`, etc.
  - Faixa preta no rodapé com a legenda em PT-BR:
    - Ex.: `um homem está sentado em frente ao computador`.

- **Uso sugerido:**
  - Demonstração visual em apresentações.
  - POC com clientes.
  - Validação qualitativa das saídas do modelo.

---

### 4.2. CSV Detalhado (`resultado.csv`)

Cada linha representa a análise de **uma pessoa em um frame**.

Campos principais:

- `frame`: número do frame no vídeo (int).
- `face_id`: id numérico da pessoa (ex.: 1, 2, 3…).
- `person_label`: rótulo amigável (`pessoa1`, `pessoa2`, ...).
- `emotion`: emoção em PT-BR (ex.: `Alegre`, `Triste`, `Neutro`).
- `score`: confiança da emoção (float).
- `caption`: legenda da cena em PT-BR (ex.: `uma mulher está falando ao telefone`).

**Exemplo de linha:**

```text
frame=120, face_id=1, person_label=pessoa1, emotion=Alegre, score=0.92, caption="um homem está digitando no computador"
