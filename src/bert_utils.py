import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertModel, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup

MAX_LEN = 128
BATCH_SIZE = 32

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
loss_fn = nn.BCEWithLogitsLoss()
DEVICE_IN_USE = ''

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
    DEVICE_IN_USE = 'cuda'

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    DEVICE_IN_USE = 'cpu'

# Create the BertClassifier class
class BertClassifier(nn.Module):
    """
    Bert Model for classification Tasks.
    """
    def __init__(self, freeze_bert=False, dataset="sportsett", num_classes=3):
        """
        @param   bert: a BertModel object
        @param   classifier: a torch.nn.Module classifier
        @param   freeze_bert (bool): Set `False` to fine_tune the Bert model
        """
        super(BertClassifier,self).__init__()
        # Specify hidden size of Bert, hidden size of our classifier, and number of labels
        self.num_classes = num_classes
        D_in, H, D_out = 768, 30, num_classes
        self.dataset = dataset

        if 'sportsett' in self.dataset or 'mlb' in self.dataset:
            print(f'\n\nIt"s {self.dataset} dataset. So loading finetuned RoBERTa model\n\n')
            self.bert = RobertaModel.from_pretrained(f'./{self.dataset}/roberta-finetuned')
        else:
            print(f'\n\nIt"s {self.dataset} dataset. So loading pre-trained RoBERTa model\n\n')
            self.bert = RobertaModel.from_pretrained('roberta-base')

        # self.bert = RobertaModel.from_pretrained('roberta-base')
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        self.classifier = nn.Sequential(
                            nn.Linear(D_in, H),
                            nn.ReLU(),
                            nn.Linear(H, D_out))
        self.sigmoid = nn.Sigmoid()
        # Freeze the Bert Model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self,input_ids,attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        outputs = self.bert(input_ids=input_ids,
                           attention_mask = attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:,0,:]
        
        # Feed input to classifier to compute logits
        logit = self.classifier(last_hidden_state_cls)
        
        # logits = self.sigmoid(logit)
        
        return logit


def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # create empty lists to store outputs
    input_ids = []
    attention_masks = []
    
    #for every sentence...    
    for sent in data:
        # 'encode_plus will':
        # (1) Tokenize the sentence
        # (2) Add the `[CLS]` and `[SEP]` token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map tokens to their IDs
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text = sent,
            add_special_tokens = True,         #Add `[CLS]` and `[SEP]`
            max_length= MAX_LEN  ,             #Max length to truncate/pad
            pad_to_max_length = True,          #pad sentence to max length 
            return_attention_mask= True        #Return attention mask 
        )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        
    #convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    
    return input_ids,attention_masks

def initialize_model(train_dataloader, epochs=4, dataset='sportsett', num_classes=3):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False, dataset=dataset, num_classes=num_classes)
    # bert_classifier = BertClassifier(freeze_bert=True, dataset=dataset, num_classes=num_classes)
    
    # bert_classifier.to(device)
    bert_classifier.cuda()
    
    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                     lr=5e-5, #Default learning rate
                     eps=1e-8 #Default epsilon value
                     )
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=0, # Default value
                                              num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train_model(model, optimizer, scheduler, train_dataloader, 
                val_dataloader=None, epochs=4, path='./sportsett/output/models/multilabel_bert.pt', 
                evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # Put the model into the training mode
        model.cuda()
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            # b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            b_input_ids, b_attn_mask, b_labels = tuple(t.cuda() for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels.float())
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20--50000 batches
            if (step % 50000 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")
    torch.save(model.state_dict(), path)

def accuracy_thresh(y_pred, y_true, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: 
        y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.byte()).float().mean().item()

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.cuda()
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        # b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        b_input_ids, b_attn_mask, b_labels = tuple(t.cuda() for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels.float())
        val_loss.append(loss.item())

        # Get the predictions
        #preds = torch.argmax(logits, dim=1).flatten()
        
        # Calculate the accuracy rate
        #accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        accuracy = accuracy_thresh(logits.view(-1,6),b_labels.view(-1,6))
        
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    # if DEVICE_IN_USE == 'gpu':
    model.cuda()
    model.eval()

    all_logits = []

    # For each batch in our test set...
    ctr = 0
    for batch in test_dataloader:
        ctr += 1
        # Load batch to GPU
        # b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        b_input_ids, b_attn_mask = tuple(t.cuda() for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    # print(ctr)

    # Apply softmax to calculate probabilities
    #probs = F.softmax(all_logits, dim=1).cpu().numpy()
    probs = all_logits.sigmoid().cpu().numpy()
    
    return probs

def train_bert_multilabel_classif(train_x, train_y, num_epochs=1, dataset='sportsett', num_classes=3, 
                                path=f'./sportsett/output/models/multilabel_bert.pt'):
    """
    Train the BERT model on the training set.
    """

    train_inputs, train_masks = preprocessing_for_bert(train_x)
    train_labels = torch.tensor(train_y)
    train_data = TensorDataset(train_inputs,train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    set_seed(42)
    bert_classifier, optimizer, scheduler = initialize_model(train_dataloader, epochs=num_epochs, dataset=dataset, num_classes=num_classes)

    return train_model(bert_classifier, optimizer, scheduler, train_dataloader, epochs=num_epochs, path=path)

def predict_bert_multilabel_classif(test_x, pred_probs=False, dataset='sportsett', num_classes=3, 
                                    path=f'./sportsett/output/models/multilabel_bert.pt', ):
    
    test_inputs, test_masks = preprocessing_for_bert(test_x)
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

    pred_model = BertClassifier(dataset=dataset, num_classes=num_classes)
    pred_model.load_state_dict(torch.load(path))
    print(f'RoBERTa model loaded')

    probs = bert_predict(pred_model, test_dataloader)
    pred_y = np.where(probs > 0.5, 1, 0)
    
    return pred_y if not pred_probs else probs
