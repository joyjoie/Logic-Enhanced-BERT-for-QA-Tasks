# CS135-Final-Project

## Scope
This project aims to develop a model that utilizes formal semantic representation (logical forms) to improve the model's capability for QA tasks. The model is designed to:
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
- Performing forward and backward passes.
- Updating the model parameters using the AdamW optimizer.

### Code Structure

The code is organized into several key sections, each responsible for a specific task in the question answering (QA) pipeline.
- The BERT tokenizer (AutoTokenizer) is used to tokenize input sequences such as the natural language question, logical form, and context.
- The AutoModelForQuestionAnswering class from Hugging Face is utilized to load the pre-trained BERT model for question answering tasks.
- A custom model class (NaturalLanguageQA for baseline, and LogicalFormQA for experiments) extends torch.nn.Module to integrate the BERT model with a cross-attention mechanism (specifically for experiments), allowing the model to attend to both the context and logical form.
- The tokenize_data function handles tokenization of the context, logical form, and question using the tokenizer. It also ensures that answers are converted into start and end positions.
- A custom QADataset class is created to hold tokenized examples, and a DataLoader is used for batching the data during training.
- The find_answer_positions function searches for the start and end indices of the answer in the context.
- The training loop iterates through the dataset, computing loss (cross-entropy) for each batch, performing backpropagation, and updating model weights using the AdamW optimizer.
- Errors during batch processing (e.g., invalid answers) are caught, and those batches are skipped without interrupting the training process.

### Execution

