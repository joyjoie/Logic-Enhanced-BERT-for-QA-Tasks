# CS135-Final-Project

## Scope
This project aims to investigate the impact of integrating formal semantic representations, specifically First Order Logical forms, into a Question Answering (QA) model, and to assess whether this integration improves or hinders the model's performance on typical QA tasks. The approach combines traditional natural language processing (NLP) with formal semantic reasoning to potentially enhance the model's ability to understand and process complex queries and contexts. 
The model is designed to:
- Tokenize and encode the natural language questions and/or their corresponding logical forms (first-order logic representations of the questions).
  - The integration of logical forms aims to possibly augment the model's understanding of the question by providing a formal semantic interpretation, alongside the usual     natural language input.
  - The input natural languagequestions will be tokenized using the BERT tokenizer, which will convert the text into tokens that the model can process. These tokens are then encoded into numerical representations (embeddings) that capture the semantic meaning of the question.
  - Each question also has a corresponding logical form, which is a formal representation of the question’s semantics. The logical form is also tokenized and encoded in the same way as the question.
- Perform cross-attention between the encoded question and context.
  - Cross-attention is designed to facilitate the interaction between different input modalities (question and context), ensuring that the model focuses on the most relevant information in the context when predicting the answer.
  - The context (passage of text that provides the potential answer) is also encoded using the same pre-trained BERT model. The model’s hidden states, representing the encoded contextual information, are passed into the attention mechanism.
  - A multi-head attention layer is introduced to enable the question to cross-reference the context. This mechanism allows the model to selectively focus on parts of the context that are most relevant to the question.
  - The output of the attention mechanism is combined with the context, producing a fused representation that includes both question and context-specific information. This is then used to predict the answer.
- Predict the start and end positions of the answer in the context.
  - After the fusion of the question and context (via the cross-attention mechanism), a Question Answering head (linear layer) is applied to predict the span of the answer. The output of this layer consists of two logits: one for the start position and one for the end position.
  - During training, the model computes the loss by comparing the predicted start and end positions with the actual positions of the answer in the context. This helps the model adjust its weights to improve its performance over time.

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

We are using the selected subset (first 5511 rows) from the TriviaQA dataset (https://huggingface.co/datasets/lucadiliello/triviaqa). Each question is used as part of the prompt and converted to the first-order logic form. (train-5117, eval-394)

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

## Execution

The code is designed to run in a Jupyter Notebook environment.  
Before running any code, ensure that all dependencies are installed.  
You can install the necessary packages directly within the notebook using the following code block:  
```!pip install torch transformers tqdm datasets pandas zhipuai```

Each code cell should be executed sequentially to ensure that the model setup, dataset preparation, and training occur in the correct order. Once trained, the model can be used for inference, and you can tweak various parameters to explore the model's behavior.

To prepare the dataset, execute the code cell inside the `data_preparation.ipynb`, load the data from TriviaQA dataset, then prompt the LLM to get the formal semantic representation and save them to a csv file.

To train the baseline model, execute the code cell inside the `baseline_qa.ipynb` sequentially, first load the train dataset from the csv file, intialize the custom model and execute the training loop, load the evaluate dataset, and execute the evaluation loop.

To perform the experiment 1 (train the BERT with only the formal semantic representations (logic forms)), execute the code cell inside the `exp1_logical.ipynb` sequentially, first load the train dataset from the csv file, intialize the custom model and execute the training loop, load the evaluate dataset, and execute the evaluation loop.

To perform the experiment 2 (train the BERT with the combination of natural language and formal semantic representations), execute the code cell inside the `exp2_logical_natural.ipynb` sequentially, first load the train dataset from the csv file, intialize the custom model and execute the training loop, load the evaluate dataset, and execute the evaluation loop.
