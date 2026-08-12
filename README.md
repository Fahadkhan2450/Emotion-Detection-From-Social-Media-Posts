# Emotion-Detection-From-Social-Media-Posts
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

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/4eaa1a34a091e563ad92a90ee0673a4e6b489251/images/RNN%20ACC%20Graph.jpeg)

![image alt](https://github.com/Fahadkhan2450/Emotion-Detection-From-Social-Media-Posts/blob/f27c893201efc24264a9ef1ef8f30c055ecec123/images/RNN%20Loss%20Graph.jpeg)

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

(Bilstm image ----------------)

# Comparing the Recurrent Architectures
With all four recurrent variants trained under comparable conditions, their results were assembled into a single comparison to evaluate whether each architectural change — bidirectionality, LSTM memory gating, or their combination — translated into a measurable improvement.

(Table image ----------)

#DistilBERT
Following the recurrent-model experiments, the project extended to a transformer-based approach using DistilBERT — a smaller, faster distillation of BERT that retains strong contextual language understanding at a fraction of the computational cost.
The key distinction between the recurrent models and DistilBERT lies in how text is represented. Where the RNN and LSTM models depended on a Keras tokenizer to build a vocabulary and manually pad integer sequences, DistilBERT uses its own pretrained tokenizer, which already encodes the vocabulary and tokenization strategy the model was originally trained with — removing the need to build a vocabulary from scratch.

(Architect image -------------------)

The pretrained model was loaded and fine-tuned on the project's seven target classes as follows:
from transformers import DistilBertForSequenceClassification
 
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(class_names)
)


Dynamic padding was used during training, so each batch was padded only to the length of its longest sequence rather than to the global maximum — meaningfully reducing memory overhead compared with uniform padding across the full dataset.
Fine-tuning used a learning rate of 2e-5, a batch size of 16, weight decay for regularization, a linear learning-rate scheduler, mixed-precision training, and validation performed after every epoch.


(loss image ________---------------)


(Acuracy image ------------)


# Evaluation and Results
Because the task involves seven classes of varying difficulty, evaluation deliberately went beyond accuracy alone. Accuracy reports the overall proportion of correctly classified samples, but can obscure how a model performs on harder or less-represented categories. Precision measures how many of a class's predicted samples were correct, recall measures how many of a class's true samples were successfully identified, and F1-score balances the two — particularly valuable when both false positives and false negatives carry meaningful consequences, as they do in a mental-health-related classification task.
A full classification report was generated for each model to examine performance at the level of individual classes.

# Final DistilBERT Classification Report

(image ----------------------)





























