# monolingual_hindi_emotion_analysis
# Bhaav Emotion Analysis

This Jupyter Notebook implements an emotion analysis pipeline for Hindi text using IndicBERT embeddings and a deep learning classifier. 

## Overview

The notebook performs the following steps:

1. **Data Loading and Preprocessing:** Loads the Bhaav dataset, handles class imbalance using hybrid resampling (oversampling and undersampling), and preprocesses the text data.
2. **Tokenization:** Tokenizes the sentences using the IndicBERT tokenizer.
3. **IndicBERT Embedding Generation:** Generates sentence embeddings using the pre-trained IndicBERT model.
4. **Model Training:** Trains a deep learning classifier (a multi-layer perceptron with batch normalization and dropout) on the IndicBERT embeddings.
5. **Evaluation:** Evaluates the trained model on a test set using metrics like accuracy, F1-score, and confusion matrix.
6. **Inference:** Provides a function to predict emotions for new Hindi sentences using the trained model and IndicBERT embeddings.
7. **Fine-tuning with IndicBERT:** Includes code (commented out) for fine-tuning the IndicBERT model itself for emotion classification.


## Dependencies

The notebook requires the following libraries:

- `pandas`
- `re`
- `torch`
- `numpy`
- `transformers`
- `sklearn`
- `inltk`
- `fasttext`
- `matplotlib`
- `seaborn`
- `wandb` (optional, for Weights & Biases logging)


## Usage

1. Upload the notebook to Google Colab.
2. Install the required libraries using the provided `pip install` commands.
3. Upload the Bhaav dataset file (`Bhaav-Dataset(1).xlsx`).
4. Run all the cells in the notebook.


## Results

The trained emotion classifier achieves an accuracy of around 77% on the test set. The confusion matrix and classification report provide detailed insights into the model's performance.

## Notes

- The fine-tuning code for IndicBERT is currently commented out. If you want to fine-tune the model, you need to uncomment the relevant code and adjust the parameters accordingly.
- The notebook includes examples of inference using pre-trained IndicBERT embeddings and the trained classifier.
- Consider using a GPU runtime in Google Colab for faster training and inference.
