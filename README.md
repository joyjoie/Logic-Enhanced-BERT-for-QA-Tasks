# CS135-Final-Project

## Scope
The scope of this project is to develop a QA model that can handle both natural language questions and logical forms. The model is designed to:
- Tokenize and encode natural language questions and logical forms.
- Perform cross-attention between the encoded question and context.
- Predict the start and end positions of the answer in the context.

## Implementation
### Model Architecture

The model is built using the `transformers` library and consists of the following components:
- **Base Model**: A pre-trained BERT model for encoding the input sequences.
- **Cross-Attention Layer**: A multi-head attention layer that allows the logical form to attend to the context.
- **QA Head**: A linear layer that predicts the start and end positions of the answer.

### Data Preparation

The data preparation involves:
- Tokenizing the natural language questions, logical forms, and contexts.
- Finding the start and end positions of the answers in the context.
- Creating a custom dataset and dataloader for training.

We are using the selected subset (first 5511 rows) from the TriviaQA dataset (https://huggingface.co/datasets/lucadiliello/triviaqa). Each question is used as part of the prompt and converted to the first-order logic form.

### Training

The training loop involves:
- Moving the model and data to the GPU (if available).
- Performing forward and backward passes.
- Updating the model parameters using the AdamW optimizer.

### Code Structure
