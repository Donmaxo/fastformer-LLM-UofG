import pandas as pd
from tnlrv3.GraphFormers.src.models.tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from tnlrv3.GraphFormers.src.models.tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer
from tnlrv3.GraphFormers.src.models.tnlrv3.config import TuringNLRv3ForSeq2SeqConfig
from transformers import BertConfig
import torch
from torch.utils.data import Dataset, DataLoader
from claude_opus_fastformer import NewsRecommendationModel, FastformerEncoder
from torch.optim import Adam
import time
from tqdm import tqdm
import zipfile
import os
import argparse
import numpy as np

# Set the number of epochs
num_epochs = 5
MAX_USER_HISTORY = 50
MAX_IMPRESSION_NEWS = 50
BATCH_SIZE = 64
dataset_type = "large"

# Set the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to train and validation dataset files
news_file_path_train = f'/mnt/c/Users/maxdo/Documents/University/data/{dataset_type}/train/news.tsv'
behaviors_file_path_train = f'/mnt/c/Users/maxdo/Documents/University/data/{dataset_type}/train/behaviors.tsv'
news_file_path_val = f'/mnt/c/Users/maxdo/Documents/University/data/{dataset_type}/valid/news.tsv'
behaviors_file_path_val = f'/mnt/c/Users/maxdo/Documents/University/data/{dataset_type}/valid/behaviors.tsv'

# Path to the UNILM model directory
model_path = "/mnt/c/Users/maxdo/Documents/University/data/unilm/minilm"


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--test_path', type=str, help='Path to the test datasets folder')
args = parser.parse_args()

# Load the TuringNLRv3 model and tokenizer
config = TuringNLRv3ForSeq2SeqConfig.from_pretrained(model_path)
tokenizer = TuringNLRv3Tokenizer.from_pretrained(model_path)
tokenizer_model = TuringNLRv3ForSequenceClassification.from_pretrained(model_path, config=config)
tokenizer_model = tokenizer_model.to(device)

# Load the Fastformer model configuration
ff_config = BertConfig.from_json_file('fastformer_claude_opus.json')

# Create an instance of the Fastformer model for generating news embeddings
ff_model = FastformerEncoder(ff_config, emb_dim=ff_config.hidden_size)
ff_model = ff_model.to(device)

def generate_embeddings(texts, tokenizer, model, ff_model, max_length=config.hidden_size, batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i:i+batch_size]
        preprocessed_texts = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        preprocessed_texts = preprocessed_texts.to(device)
        with torch.no_grad():
            attention_mask = preprocessed_texts['attention_mask']
            outputs = model(**preprocessed_texts)
            batch_embeddings = outputs[0].mean(dim=1)
            batch_embeddings = ff_model(batch_embeddings.unsqueeze(1), attention_mask[:, :1], -1)
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings)

def calculate_accuracy(scores, targets):
    scores = scores.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    predicts = scores > 0.5
    accuracy = (predicts == targets).mean()
    return accuracy

class MINDDataset(Dataset):
    def __init__(self, behaviors_df, news_embedding_dict, is_test=False):
        self.behaviors_df = behaviors_df
        self.news_embedding_dict = news_embedding_dict
        self.is_test = is_test

    def __len__(self):
        return len(self.behaviors_df)

    def __getitem__(self, idx):
        row = self.behaviors_df.iloc[idx]
        
        if pd.isna(row['History']):
            user_history = []
        else:
            user_history = row['History'].split()
        
        if pd.isna(row['Impressions']):
            impressions = []
        else:
            impressions = row['Impressions'].split()
        
        user_history_embds = [self.news_embedding_dict[news_id] for news_id in user_history if news_id in self.news_embedding_dict]
        impression_embds = []
        labels = []
        for impression in impressions:
            if self.is_test:
                news_id = impression
            else:
                news_id, label = impression.split('-')
                labels.append(int(label))
            if news_id in self.news_embedding_dict:
                impression_embds.append(self.news_embedding_dict[news_id])
        
        user_history_embds = user_history_embds[:MAX_USER_HISTORY]
        user_history_embds += [torch.zeros(ff_config.hidden_size)] * (MAX_USER_HISTORY - len(user_history_embds))
        impression_embds = impression_embds[:MAX_USER_HISTORY]
        impression_embds += [torch.zeros(ff_config.hidden_size)] * (MAX_USER_HISTORY - len(impression_embds))
        if not self.is_test:
            labels = labels[:MAX_USER_HISTORY]
            labels += [0] * (MAX_USER_HISTORY - len(labels))

        user_history_embds = torch.stack(user_history_embds)
        impression_embds = torch.stack(impression_embds)
        if self.is_test:
            impression_id = row['ImpressionID']
            return user_history_embds, impression_embds, impression_id
        else:
            labels = torch.tensor(labels)
            return user_history_embds, impression_embds, labels
    
def run_predictions(model, news_df, behaviors_df, output_path):
    # Generate news embeddings for the test dataset
    print("Generating test news embeddings...")
    start_time = time.time()
    news_titles = news_df['Title'].tolist()
    news_embeddings = generate_embeddings(news_titles, tokenizer, tokenizer_model, ff_model, max_length=512)
    news_embedding_dict_test = dict(zip(news_df['NewsID'], news_embeddings))
    end_time = time.time()
    duration = end_time - start_time
    print(f"Test news embeddings generated in {duration:.2f} seconds")
    
    test_dataset = MINDDataset(behaviors_df, news_embedding_dict_test, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    group_impr_indexes = []
    group_preds = []

    with torch.no_grad():
        for user_history_embds, impression_embds, impression_ids in tqdm(test_loader, desc="Running Predictions"):
            user_history_embds = user_history_embds.to(device)
            impression_embds = impression_embds.to(device)

            user_embds = model(user_history_embds, None, None)
            user_embds = user_embds.detach().cpu().numpy()
            impression_embds = impression_embds.detach().cpu().numpy()

            scores = np.dot(impression_embds, user_embds.T)

            for impr_id, score in zip(impression_ids, scores):
                group_impr_indexes.append(impr_id)
                group_preds.append(score)

    with open(os.path.join(output_path, 'prediction.txt'), 'w') as f:
        for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds), desc="Writing Predictions"):
            impr_index += 1
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
            f.write(' '.join([str(impr_index), pred_rank]) + '\n')

    f = zipfile.ZipFile(os.path.join(output_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
    f.write(os.path.join(output_path, 'prediction.txt'), arcname='prediction.txt')
    f.close()

    print("Predictions written to file")

if args.test_path:
    # Paths to test dataset files
    test_news_file_path = f'{args.test_path}/news.tsv'
    test_behaviors_file_path = f'{args.test_path}/behaviors.tsv'

    # Load and preprocess test dataset
    test_news_df = pd.read_csv(test_news_file_path, sep='\t', header=None, names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
    test_behaviors_df = pd.read_csv(test_behaviors_file_path, sep='\t', header=None, names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

    # Initialize the Fastformer model
    nr_model = NewsRecommendationModel(ff_config, news_embedding_dim=ff_config.hidden_size, num_classes=MAX_IMPRESSION_NEWS, device=device)
    nr_model = nr_model.to(device)

    # Load the trained model
    nr_model.load_state_dict(torch.load("trained-models/fastformer_model_final.pth"))
    nr_model.eval()

    # Run predictions on the test dataset
    run_predictions(nr_model, test_news_df, test_behaviors_df, "predictions/")

else:
    # Load and preprocess train and validation datasets
    news_df_train = pd.read_csv(news_file_path_train, sep='\t', header=None, names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
    behaviors_df_train = pd.read_csv(behaviors_file_path_train, sep='\t', header=None, names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
    news_df_val = pd.read_csv(news_file_path_val, sep='\t', header=None, names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
    behaviors_df_val = pd.read_csv(behaviors_file_path_val, sep='\t', header=None, names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

    # Process news titles and generate embeddings
    print("Generating news embeddings...")
    start_time = time.time()
    news_titles_train = news_df_train['Title'].tolist()
    news_embeddings_train = generate_embeddings(news_titles_train, tokenizer, tokenizer_model, ff_model, max_length=512)
    news_embedding_dict_train = dict(zip(news_df_train['NewsID'], news_embeddings_train))
    end_time = time.time()
    duration = end_time - start_time
    print(f"News embeddings generated in {duration:.2f} seconds")

    # Create DataLoader
    train_dataset = MINDDataset(behaviors_df_train, news_embedding_dict_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = MINDDataset(behaviors_df_val, news_embedding_dict_train)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the Fastformer model
    nr_model = NewsRecommendationModel(ff_config, news_embedding_dim=ff_config.hidden_size, num_classes=MAX_IMPRESSION_NEWS, device=device)
    nr_model = nr_model.to(device)

    # Set up the optimizer
    optimizer = Adam(nr_model.parameters(), lr=1e-5)

    # Training loop
    for epoch in range(num_epochs):
        nr_model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for user_history_embds, impression_embds, labels in progress_bar:
            user_history_embds = user_history_embds.to(device)
            impression_embds = impression_embds.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss, scores = nr_model(user_history_embds, impression_embds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=avg_loss)

        # Evaluate on the validation set
        nr_model.eval()
        val_accuracy = 0
        with torch.no_grad():
            for user_history_embds, impression_embds, labels in val_loader:
                user_history_embds = user_history_embds.to(device)
                impression_embds = impression_embds.to(device)
                labels = labels.to(device)

                _, scores = nr_model(user_history_embds, impression_embds, labels)
                val_accuracy += calculate_accuracy(scores, labels)

        val_accuracy /= len(val_loader)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
        # Save the model after every epoch
        torch.save(nr_model.state_dict(), f"trained-models/fastformer_model_epoch_{epoch+1}.pth")

    # Save the final model
    torch.save(nr_model.state_dict(), "trained-models/fastformer_model_final.pth")