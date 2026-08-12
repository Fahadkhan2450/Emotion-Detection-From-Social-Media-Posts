# Comparative Evaluation of RNN, LSTM, BiLSTM and Transformer Models for Mental Health Text Classification
A Progressive Comparison of Recurrent and Transformer-Based Architectures for Multiclass Mental-Health Text Classification

# ABSTRACT
Social media has become one of the richest sources of unstructured text reflecting how people feel, think, and cope on a day-to-day basis. Buried within this volume of informal writing are signals relevant to emotional and mental health states , signals that are far too extensive to review manually, but well suited to automated analysis through Natural Language Processing (NLP) and deep learning.
This project develops a multiclass text classification system trained on the Sentiment Analysis for Mental Health dataset from Kaggle, which organizes social media statements into seven categories: Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, and Suicidal.
Rather than jumping directly to a state-of-the-art architecture, the project was deliberately structured as an incremental progression. It begins with a Simple RNN baseline, which is then refined through systematic tuning of learning rate, dropout, hidden-unit count, batch size, and optimization strategy, supported by Early Stopping, ReduceLROnPlateau, and Model Checkpointing callbacks. Building on these findings, LSTM and Bidirectional LSTM (BiLSTM) models were introduced to better capture long range dependencies in text, and the investigation culminates in fine tuning DistilBERT, a pretrained transformer model, using its native tokenizer and sequence classification architecture.
Every model is evaluated on a consistent set of metrics: accuracy, precision, recall, F1-score, loss, confusion matrices, and multiclass ROC-AUC curves — alongside training and validation curves used to diagnose learning behavior and overfitting.

# Introduction
As more of everyday communication moves online, social media posts have become an increasingly valuable window into how people express emotion, describe their experiences, and process difficult moments. This language, however, is rarely tidy: it is informal, often abbreviated, inconsistent in structure, and heavily dependent on context ; conditions under which traditional rule based text analysis tends to break down.
Deep learning offers a more resilient alternative, since it learns patterns directly from data rather than relying on hand crafted rules. Recurrent Neural Networks (RNNs) were an early and natural fit for this kind of sequential data, processing text word by word while carrying forward information from what came before. Yet a basic RNN struggles to retain information across longer sequences , a limitation that becomes especially apparent as sentence length grows.
To address this limitation methodically rather than by simply reaching for a larger model, the project follows a staged development path. A Simple RNN establishes the baseline.
LSTM was then introduced for its ability to preserve relevant information over much longer spans of text through its internal gating mechanism, and a Bidirectional LSTM extended this by combining long-range memory with bidirectional context. Finally, the investigation shifts from purely recurrent architectures to a transformer based approach using DistilBERT, allowing a direct comparison between traditional sequential models and a modern pretrained language model.
![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/4eaa1a34a091e563ad92a90ee0673a4e6b489251/images/pipe%20line%202.jpeg)


# Dataset and Preprocessing
The project uses the publicly available Kaggle dataset suchintikasarkar/sentiment-analysis-for-mental-health, retrieved via KaggleHub. The Combined Data.csv file provided two fields of direct relevance: statement, the raw text supplied to the model, and status, the corresponding mental-health-related category.
The seven target categories are Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, and Suicidal.
Prior to training, duplicate records were removed and rows with missing values in either required column were dropped. All text was cast to string type to guarantee compatibility with the tokenizer, and the categorical status labels were converted into numerical form using a LabelEncoder.
For the recurrent models, the dataset was partitioned into training, validation, and test sets using stratified sampling, which preserves the relative proportions of all seven classes across each split. The training set was used to fit model parameters, the validation set to monitor performance and guide tuning decisions during training, and the test set was reserved exclusively for final, unbiased evaluation.

Training Set  →  used for learning
Validation Set  →  used for monitoring and tuning
Testing Set  →  used only for final evaluation

# Model Development
Model development proceeded in stages, beginning with a Simple RNN to establish how a basic recurrent architecture performs on this dataset before layering in additional complexity.
For the RNN and LSTM models, a Keras tokenizer was fitted exclusively on the training text and then applied to convert each statement into a sequence of integers. Since statements vary considerably in length, all sequences were padded to a fixed length so they could be batched together; sequences exceeding this length were truncated, while shorter ones were padded.
A maximum vocabulary size of approximately 30,000 words and a maximum sequence length of 200 tokens were used throughout the recurrent-model experiments (MAX_WORDS = 30000, MAX_LEN = 200).

# Simple RNN
The initial architecture consisted of an embedding layer, a recurrent layer, dropout for regularization, dense layers, and a final softmax layer producing probabilities across the seven classes.


![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/ce6842c25ea3ccfc14f4a2e6836a6ea049b6e433/images/RNN%20archt.jpeg)

# Paramters Involved in RNN

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/f27c893201efc24264a9ef1ef8f30c055ecec123/images/RNN%20Training%20params.jpeg)

This baseline model was able to fit the training data reasonably well,but validation accuracy fluctuated rather than improving consistently  an early indication that additional training alone would not resolve the model's limitations, and that both the architecture and training procedure needed refinement.

![image alt](https://github.com/Fahadkhan2450/deep-learning-mental-health-classification/blob/adb530d83c95725ebd0cf0bd54b8a0a98c453ad8/images/R%20acc%20%2Closs.jpeg)


# Class based RNN Scores

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/74e113298364fd1288bfcd60ee23d242fc314c91/images/RNN%20Class%20based%20scores.jpeg)


# LSTM
A conventional RNN, struggles to retain useful information across long stretches of text. LSTM addresses this through an internal memory cell and a system of gates that regulate which information is retained, updated, or discarded over time. The same dataset and evaluation procedure used for the RNN experiments were applied to the LSTM architecture to allow a direct, like for like comparison.

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/74e113298364fd1288bfcd60ee23d242fc314c91/images/LSTM%20Atcht.jpeg)

# LSTM Params Involved 

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/74e113298364fd1288bfcd60ee23d242fc314c91/images/LSTM%20Params.jpeg)


# LSTM Results 

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/74e113298364fd1288bfcd60ee23d242fc314c91/images/LSTM%20Conf%20Matrix.jpeg)

# LSTM Classification Report
![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/e276efbd611adc967246977860bb8a2d984aa537/images/LST%20Class%20report.jpeg)


# Bidirectional LSTM (BiLSTM)

Extending the LSTM in the same way the earlier RNN was extended, a Bidirectional LSTM was implemented to process text in both directions — but with each direction now handled by a full LSTM cell rather than a basic recurrent unit. Additional regularization, alongside further adjustments to learning rate and callback configuration, was applied at this stage based on validation performance observed during the earlier experiments.
![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/5874852a6957dd15781a0693a90b1a5633347ed1/images/BI%20LSTM%20Params.jpeg)

The final BiLSTM model consisted of a total of [X] trainable parameters. On the held-out test set, the model achieved a test accuracy of 75%, with the corresponding training and validation accuracy curves shown below.

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/5874852a6957dd15781a0693a90b1a5633347ed1/images/Bi%20LSTM%20accuracy.jpeg)

To further evaluate the model's ability to distinguish between the seven classes, a multiclass ROC-AUC analysis was performed using a One-vs-Rest approach, producing an individual ROC curve for each category.

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/5874852a6957dd15781a0693a90b1a5633347ed1/images/Bi%20lstm%20roc%20curve.jpeg)

# Comparing the Recurrent Architectures
With all four recurrent variants trained under comparable conditions, their results were assembled into a single comparison to evaluate whether each architectural change — bidirectionality, LSTM memory gating, or their combination — translated into a measurable improvement.

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/fd72a86572e04f7a176fb30a13709aa78b793089/images/Table%20For%20Acc%20and%20loss.jpeg)

# DistilBERT
Following the recurrent-model experiments, the project extended to a transformer-based approach using DistilBERT , a smaller, faster distillation of BERT that retains strong contextual language understanding at a fraction of the computational cost.
The key distinction between the recurrent models and DistilBERT lies in how text is represented. Where the RNN and LSTM models depended on a Keras tokenizer to build a vocabulary and manually pad integer sequences, DistilBERT uses its own pretrained tokenizer, which already encodes the vocabulary and tokenization strategy the model was originally trained with , removing the need to build a vocabulary from scratch.

![image alt](https://github.com/Fahadkhan2450/deep-learning-mental-health-classification/blob/223e4553a9e9d68b3df9afcf807c858ecbaae238/images/DISTU.jpeg)

The pretrained model was loaded and fine-tuned on the project's seven target classes as follows:
from transformers import DistilBertForSequenceClassification
 
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(class_names)
)


Dynamic padding was used during training, so each batch was padded only to the length of its longest sequence rather than to the global maximum — meaningfully reducing memory overhead compared with uniform padding across the full dataset.
Fine-tuning used a learning rate of 2e-5, a batch size of 16, weight decay for regularization, a linear learning-rate scheduler, mixed-precision training, and validation performed after every epoch.


![image alt](https://github.com/Fahadkhan2450/deep-learning-mental-health-classification/blob/526957010797218426733d55cd14e8fcdbd1195b/images/Dist%20p.jpeg)


Evaluation and Results
Because the task involves seven classes of varying difficulty, evaluation deliberately went beyond accuracy alone. Accuracy reports the overall proportion of correctly classified samples, but can obscure how a model performs on harder or less-represented categories. Precision measures how many of a class's predicted samples were correct, recall measures how many of a class's true samples were successfully identified, and F1-score balances the two — particularly valuable when both false positives and false negatives carry meaningful consequences, as they do in a mental-health-related classification task.
A full classification report was generated for each model to examine performance at the level of individual classes.

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/e981317f1abc062abb077d9bb085feb80bb77d6e/images/Distill%20Classification%20Report.jpeg)

Finally, a multiclass ROC-AUC analysis was carried out using a One-vs-Rest strategy, in which each class is treated in turn as the positive class against all others. This produces an individual ROC curve per category and offers a more granular view of the model's ability to discriminate between classes than accuracy alone can provide.
![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/748f53a2314ac4a690cf7ba63463df1b19ef97aa/images/Distill%20ROc%20Curve.jpeg)

# Final Performance Table 

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/744d40ec032d21b0524e010f7969ebec22d3b4d1/images/FInal%20Table.jpeg)


# Conclusion
This project examined several complementary approaches to classifying social media posts into mental-health-related categories, deliberately favoring a progressive, staged methodology over a single-model solution in order to understand how each architectural and training decision affected performance.
The Simple RNN served as an initial baseline before being extended into a Bidirectional RNN capable of drawing on context from both directions of a sentence. From there, systematic tuning of learning rate, dropout, recurrent units, and batch size — supported by Early Stopping, ReduceLROnPlateau, and Model Checkpointing — improved training stability and reduced overfitting.
LSTM was then introduced for its capacity to retain relevant information across longer sequences, and Bidirectional LSTM combined this long-range memory with bidirectional context. The investigation concluded with DistilBERT, a pretrained transformer whose attention-based architecture stands in direct contrast to the recurrent models that preceded it — enabling a clear, structured comparison between traditional sequential deep learning and modern pretrained language models.
Taken together, the training curves, loss curves, class-level metrics, confusion matrices, and ROC-AUC analysis presented above offer a detailed picture of how each architecture performs on this dataset.
It should be emphasized that these models are intended strictly for text classification and research purposes. A model prediction is not, and should never be treated as, a medical or psychological diagnosis.

# Future Work
Several directions could extend this work further. Additional transformer architectures — including BERT, RoBERTa, ALBERT, and DeBERTa — could be evaluated under the same framework to test whether DistilBERT's efficiency comes at a meaningful cost to accuracy. Data augmentation, ensemble modeling, and explainable-AI techniques would also be natural next steps, as would validating the final model against an independent, held-out dataset to test generalization beyond the source corpus.
A further valuable direction would be a qualitative error analysis: examining misclassified posts directly to identify the linguistic patterns that make particular classes difficult to distinguish. Such analysis could reveal specific weaknesses in the current models and guide more targeted improvements in future iterations of the system.


# Refrences 
1)  U. Gupta, A. Chatterjee, R. Srikanth, and P. Agrawal, “A Sentimentand-Semantics-Based Approach for Emotion Detection in Textual Conversations,” arXiv, 2017. [Online]. Available: http://arxiv.org/abs/1707.06996
2)  Jingwen Liu, Weichang Gao, Meijun Ou, Bin Fu, Xuanbo Huang, Yisha Huan, "OnDeBERTa: An Ordered Neuron Memory Enhanced DeBERTa Model for E-commerce Review Sentiment Classification", 2026 International Conference on Computer Intelligence and Software Engineering (CICSE), pp.400-405, 2026.
3)  L. O. Oyaniyi, G. F. A. Musa and N. P. Nguyen, "Context-Aware Sentiment Classification of Parenting Narratives on AI and Social Media Using Fine-Tuned Transformers," 2026 IEEE/ACIS 24th International Conference on Software Engineering Research, Management and Applications (SERA), Towson, MD, USA, 2026, pp. 179-184, doi: 10.1109/SERA69989.2026.11618647.
4)  G. S. Prahasto and E. B. Setiawan, “Twitter Social Media-Based Sentiment Analysis Using Bi-LSTM Method With Genetic Algorithms Optimization”, khif, vol. 11, no. 1, pp. 20–26, Jul. 2025.
5)  K. A, A. B and D. G, "Maximizing Sentiment Detection Through Comprehensive Multimodal Data Fusion: Integrating CNN, RNN, LSTM," 2025 International Conference on Multi-Agent Systems for Collaborative Intelligence (ICMSCI), Erode, India, 2025, pp. 1-7, doi: 10.1109/ICMSCI62561.2025.10894282.





























