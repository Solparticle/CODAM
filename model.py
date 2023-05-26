import pandas as pd
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Read the dataset using pandas
df = pd.read_json('dev-v1.1.json')

# Preprocess the dataset
# Your preprocessing code goes here

# Split the data into questions and answers
questions = df['question'].tolist()
answers = df['answer'].tolist()

# Preprocess the questions and answers
# Your preprocessing code goes here

# Tokenize the questions and answers
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_inputs = tokenizer.batch_encode_plus(
    questions,
    answers,
    padding = True,
    truncation = True,
    max_length = 512,
    return_tensors = 'pt'
)

input_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']

# Split the data into training and testing sets
train_size = int(0.8 * len(input_ids))
train_input_ids = input_ids[:train_size]
train_attention_mask = attention_mask[:train_size]

test_input_ids = input_ids[train_size:]
test_attention_mask = attention_mask[train_size:]

# Load the BERT model for question answering
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Set up the training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    
model.train()

optimizer.zero_grad()
outputs = model(
    input_ids = train_input_ids,
    attention_mask = train_attention_mask,
    start_positions = start_positions, # Provide the correct start positions
    end_positions = end_positions # Provide the correct end positions
)

loss = outputs.loss
loss.backward()
optimizer.step()

# Evaluate the model
model.eval()

# Prompt user for a question
question = input("Enter your question: ")

# Tokenize and encode the input question
encoded_input = tokenizer.encode_plus(
    question,
    None,
    add_special_tokens = True,
    padding = True,
    truncation = True,
    max_length = 512,
    return_tensors = 'pt'
)

input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']

# Make predictions
with torch.no_grad():
outputs = model(
    input_ids = input_ids,
    attention_mask = attention_mask,
)

start_scores = outputs.start_logits
end_scores = outputs.end_logits

# Decode the predicted answer
all_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)

answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index+1])
print("Question:", question)
print("Answer:", answer)