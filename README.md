<h1 align="center">Comparative Evaluation of RNN, LSTM, BiLSTM, and Transformer Models for Mental Health Text Classification</h1>

<p align="center"><i>A Progressive Comparison of Recurrent and Transformer-Based Architectures for Multiclass Mental-Health Text Classification</i></p>

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Dataset and Preprocessing](#3-dataset-and-preprocessing)
4. [Model Development](#4-model-development)
   - [4.1 Simple RNN](#41-simple-rnn)
   - [4.2 LSTM](#42-lstm)
   - [4.3 Bidirectional LSTM (BiLSTM)](#43-bidirectional-lstm-bilstm)
   - [4.4 Comparing the Recurrent Architectures](#44-comparing-the-recurrent-architectures)
   - [4.5 DistilBERT](#45-distilbert)
5. [Evaluation and Results](#5-evaluation-and-results)
6. [Final Performance Comparison](#6-final-performance-comparison)
7. [Conclusion](#7-conclusion)
8. [Future Work](#8-future-work)
9. [References](#9-references)

---

## 1. Abstract

Social media has become one of the richest sources of unstructured text reflecting how people feel, think, and cope day to day. Buried within this volume of informal writing are signals relevant to emotional and mental health states — signals too extensive to review manually, but well suited to automated analysis through Natural Language Processing (NLP) and deep learning.

This project develops a **multiclass text classification system** trained on the *Sentiment Analysis for Mental Health* dataset from Kaggle, organizing social media statements into seven categories: **Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress,** and **Suicidal**.

Rather than jumping directly to a state-of-the-art architecture, the project was deliberately structured as an **incremental progression**:

| Stage | Architecture | Purpose |
|---|---|---|
| 1 | Simple RNN | Establish a recurrent baseline |
| 2 | LSTM | Capture longer-range dependencies via gating |
| 3 | Bidirectional LSTM | Add bidirectional context on top of LSTM memory |
| 4 | DistilBERT (fine-tuned) | Compare against a pretrained transformer |

Every model is evaluated on a consistent set of metrics — **accuracy, precision, recall, F1-score, loss, confusion matrices, and multiclass ROC-AUC curves** — alongside training/validation curves used to diagnose learning behavior and overfitting.

---

## 2. Introduction

As more everyday communication moves online, social media posts have become an increasingly valuable window into how people express emotion, describe their experiences, and process difficult moments. This language is rarely tidy — informal, often abbreviated, inconsistent in structure, and heavily context-dependent — conditions under which traditional rule-based text analysis tends to break down.

Deep learning offers a more resilient alternative, learning patterns directly from data rather than relying on hand-crafted rules. **Recurrent Neural Networks (RNNs)** were a natural early fit for sequential text, but struggle to retain information across longer sequences. To address this methodically rather than by simply reaching for a larger model, the project follows a staged path:

- A **Simple RNN** establishes the baseline.
- **LSTM** is introduced for its gated memory, preserving relevant information over longer spans.
- **Bidirectional LSTM** extends this with context from both directions of a sentence.
- **DistilBERT** shifts from purely recurrent architectures to a transformer-based approach, enabling a direct comparison against modern pretrained language models.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/4eaa1a34a091e563ad92a90ee0673a4e6b489251/images/pipe%20line%202.jpeg" width="500" alt="Project pipeline">
</p>

---

## 3. Dataset and Preprocessing

The project uses the publicly available Kaggle dataset **`suchintikasarkar/sentiment-analysis-for-mental-health`**, retrieved via KaggleHub. The `Combined Data.csv` file provides two fields of direct relevance:

| Field | Description |
|---|---|
| `statement` | Raw text supplied to the model |
| `status` | Corresponding mental-health-related category (label) |

**Target categories (7):**

| # | Category |
|---:|---|
| 1 | Anxiety |
| 2 | Bipolar |
| 3 | Depression |
| 4 | Normal |
| 5 | Personality Disorder |
| 6 | Stress |
| 7 | Suicidal |

**Preprocessing steps:**

| Step | Detail |
|---|---|
| Deduplication | Duplicate records removed |
| Missing values | Rows with missing `statement` or `status` dropped |
| Type casting | All text cast to string for tokenizer compatibility |
| Label encoding | Categorical `status` labels converted to numeric form via `LabelEncoder` |
| Splitting | Stratified sampling into train / validation / test, preserving class proportions |

| Split | Purpose |
|---|---|
| Training Set | Used for learning (fitting model parameters) |
| Validation Set | Used for monitoring and tuning during training |
| Test Set | Used only for final, unbiased evaluation |

---

## 4. Model Development

Model development proceeded in stages, beginning with a Simple RNN to establish how a basic recurrent architecture performs on this dataset before layering in additional complexity.

For the RNN and LSTM models, a Keras tokenizer was fitted **exclusively on the training text** and applied to convert each statement into a sequence of integers. Sequences were padded/truncated to a fixed length so they could be batched together.

| Hyperparameter | Value |
|---|---|
| `MAX_WORDS` (vocabulary size) | ~30,000 |
| `MAX_LEN` (sequence length) | 200 tokens |

### 4.1 Simple RNN

Architecture: embedding layer → recurrent layer → dropout (regularization) → dense layers → softmax output over 7 classes.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/ce6842c25ea3ccfc14f4a2e6836a6ea049b6e433/images/RNN%20archt.jpeg" width="500" alt="Simple RNN architecture">
</p>

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/f27c893201efc24264a9ef1ef8f30c055ecec123/images/RNN%20Training%20params.jpeg" width="500" alt="RNN training parameters">
</p>

The baseline model fit the training data reasonably well, but **validation accuracy fluctuated rather than improving consistently** — an early signal that additional training alone would not resolve the model's limitations, and that both architecture and training procedure needed refinement.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/deep-learning-mental-health-classification/blob/adb530d83c95725ebd0cf0bd54b8a0a98c453ad8/images/R%20acc%20%2Closs.jpeg" width="500" alt="RNN accuracy and loss curves">
</p>

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/74e113298364fd1288bfcd60ee23d242fc314c91/images/RNN%20Class%20based%20scores.jpeg" width="500" alt="RNN class-based scores">
</p>

### 4.2 LSTM

A conventional RNN struggles to retain useful information across long stretches of text. **LSTM** addresses this through an internal memory cell and gating mechanism that regulates which information is retained, updated, or discarded over time. The same dataset and evaluation procedure used for the RNN experiments were applied here for a direct, like-for-like comparison.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/74e113298364fd1288bfcd60ee23d242fc314c91/images/LSTM%20Atcht.jpeg" width="500" alt="LSTM architecture">
</p>

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/74e113298364fd1288bfcd60ee23d242fc314c91/images/LSTM%20Params.jpeg" width="500" alt="LSTM parameters">
</p>

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/74e113298364fd1288bfcd60ee23d242fc314c91/images/LSTM%20Conf%20Matrix.jpeg" width="500" alt="LSTM confusion matrix">
</p>

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/e276efbd611adc967246977860bb8a2d984aa537/images/LST%20Class%20report.jpeg" width="500" alt="LSTM classification report">
</p>

### 4.3 Bidirectional LSTM (BiLSTM)

Extending the LSTM the same way the earlier RNN was extended, a **Bidirectional LSTM** processes text in both directions — with each direction handled by a full LSTM cell rather than a basic recurrent unit. Additional regularization, along with further adjustments to learning rate and callback configuration, was applied based on validation performance observed during earlier experiments.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/5874852a6957dd15781a0693a90b1a5633347ed1/images/BI%20LSTM%20Params.jpeg" width="500" alt="BiLSTM parameters">
</p>

| Result | Value |
|---|---|
| Test Accuracy | **75%** |

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/5874852a6957dd15781a0693a90b1a5633347ed1/images/Bi%20LSTM%20accuracy.jpeg" width="500" alt="BiLSTM accuracy curves">
</p>

A multiclass **ROC-AUC** analysis was performed using a One-vs-Rest approach, producing an individual ROC curve per category.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/5874852a6957dd15781a0693a90b1a5633347ed1/images/Bi%20lstm%20roc%20curve.jpeg" width="500" alt="BiLSTM ROC-AUC curves">
</p>

### 4.4 Comparing the Recurrent Architectures

With all four recurrent variants trained under comparable conditions, results were assembled into a single comparison to evaluate whether each architectural change — bidirectionality, LSTM memory gating, or their combination — translated into a measurable improvement.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/fd72a86572e04f7a176fb30a13709aa78b793089/images/Table%20For%20Acc%20and%20loss.jpeg" width="500" alt="Recurrent model comparison table">
</p>

### 4.5 DistilBERT

Following the recurrent-model experiments, the project extended to a transformer-based approach using **DistilBERT** — a smaller, faster distillation of BERT that retains strong contextual language understanding at a fraction of the computational cost.

The key distinction from the recurrent models lies in text representation: rather than a Keras tokenizer building a vocabulary from scratch, DistilBERT uses its **own pretrained tokenizer**, encoding the vocabulary and tokenization strategy it was originally trained with.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/deep-learning-mental-health-classification/blob/223e4553a9e9d68b3df9afcf807c858ecbaae238/images/DISTU.jpeg" width="500" alt="DistilBERT overview">
</p>

The pretrained model was loaded and fine-tuned on the project's seven target classes:

```python
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(class_names)
)
```

**Fine-tuning configuration:**

| Parameter | Value |
|---|---|
| Tokenizer | Pretrained DistilBERT tokenizer (dynamic padding per batch) |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Regularization | Weight decay |
| Scheduler | Linear learning-rate scheduler |
| Precision | Mixed-precision training |
| Validation | Performed after every epoch |

> Dynamic padding pads each batch only to the length of its longest sequence rather than the global maximum, meaningfully reducing memory overhead compared with uniform padding.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/deep-learning-mental-health-classification/blob/526957010797218426733d55cd14e8fcdbd1195b/images/Dist%20p.jpeg" width="500" alt="DistilBERT training progress">
</p>

---

## 5. Evaluation and Results

Because the task involves seven classes of varying difficulty, evaluation deliberately went beyond accuracy alone:

| Metric | What it Measures |
|---|---|
| **Accuracy** | Overall proportion of correctly classified samples |
| **Precision** | Of predicted samples for a class, how many were correct |
| **Recall** | Of true samples for a class, how many were identified |
| **F1-score** | Harmonic balance of precision and recall — critical when both false positives and false negatives carry real consequences, as in mental-health classification |

A full classification report was generated for each model to examine performance at the individual-class level.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/e981317f1abc062abb077d9bb085feb80bb77d6e/images/Distill%20Classification%20Report.jpeg" width="500" alt="DistilBERT classification report">
</p>

A multiclass **ROC-AUC** analysis was carried out using a One-vs-Rest strategy, treating each class in turn as the positive class against all others — offering a more granular view of discriminative ability than accuracy alone.

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/748f53a2314ac4a690cf7ba63463df1b19ef97aa/images/Distill%20ROc%20Curve.jpeg" width="500" alt="DistilBERT ROC-AUC curves">
</p>

---

## 6. Final Performance Comparison

<p align="center">
  <img src="https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/744d40ec032d21b0524e010f7969ebec22d3b4d1/images/FInal%20Table.jpeg" width="500" alt="Final performance comparison table">
</p>

---

## 7. Conclusion

This project examined several complementary approaches to classifying social media posts into mental-health-related categories, deliberately favoring a **progressive, staged methodology** over a single-model solution in order to understand how each architectural and training decision affected performance.

The Simple RNN served as an initial baseline before systematic tuning of learning rate, dropout, recurrent units, and batch size — supported by Early Stopping, ReduceLROnPlateau, and Model Checkpointing — improved training stability and reduced overfitting. LSTM was then introduced for its capacity to retain relevant information across longer sequences, and Bidirectional LSTM combined this long-range memory with bidirectional context. The investigation concluded with **DistilBERT**, a pretrained transformer whose attention-based architecture stands in direct contrast to the recurrent models that preceded it, enabling a clear, structured comparison between traditional sequential deep learning and modern pretrained language models.

Taken together, the training curves, loss curves, class-level metrics, confusion matrices, and ROC-AUC analysis presented above offer a detailed picture of how each architecture performs on this dataset.

> ⚠️ **Disclaimer:** These models are intended strictly for text classification and research purposes. A model prediction is **not**, and should never be treated as, a medical or psychological diagnosis.

---

## 8. Future Work

| Direction | Description |
|---|---|
| Additional transformers | Evaluate BERT, RoBERTa, ALBERT, and DeBERTa under the same framework to test whether DistilBERT's efficiency comes at a meaningful accuracy cost |
| Data augmentation | Explore augmentation techniques to improve robustness on underrepresented classes |
| Ensemble modeling | Combine recurrent and transformer predictions for improved robustness |
| Explainable AI | Apply interpretability techniques (e.g., attention visualization, SHAP/LIME) to understand model decisions |
| External validation | Test the final model against an independent, held-out dataset to assess generalization beyond the source corpus |
| Qualitative error analysis | Examine misclassified posts directly to identify linguistic patterns that make specific classes difficult to distinguish, guiding targeted improvements |

---

## 9. References

1. U. Gupta, A. Chatterjee, R. Srikanth, and P. Agrawal, "A Sentiment-and-Semantics-Based Approach for Emotion Detection in Textual Conversations," *arXiv*, 2017. [Online]. Available: http://arxiv.org/abs/1707.06996
2. J. Liu, W. Gao, M. Ou, B. Fu, X. Huang, and Y. Huan, "OnDeBERTa: An Ordered Neuron Memory Enhanced DeBERTa Model for E-commerce Review Sentiment Classification," *2026 International Conference on Computer Intelligence and Software Engineering (CICSE)*, pp. 400–405, 2026.
3. L. O. Oyaniyi, G. F. A. Musa, and N. P. Nguyen, "Context-Aware Sentiment Classification of Parenting Narratives on AI and Social Media Using Fine-Tuned Transformers," *2026 IEEE/ACIS 24th International Conference on Software Engineering Research, Management and Applications (SERA)*, Towson, MD, USA, 2026, pp. 179–184, doi: 10.1109/SERA69989.2026.11618647.
4. G. S. Prahasto and E. B. Setiawan, "Twitter Social Media-Based Sentiment Analysis Using Bi-LSTM Method With Genetic Algorithms Optimization," *KHIF*, vol. 11, no. 1, pp. 20–26, Jul. 2025.
5. K. A, A. B, and D. G, "Maximizing Sentiment Detection Through Comprehensive Multimodal Data Fusion: Integrating CNN, RNN, LSTM," *2025 International Conference on Multi-Agent Systems for Collaborative Intelligence (ICMSCI)*, Erode, India, 2025, pp. 1–7, doi: 10.1109/ICMSCI62561.2025.10894282.

---

<p align="center"><i>Contributions, issues, and pull requests are welcome.</i></p>
