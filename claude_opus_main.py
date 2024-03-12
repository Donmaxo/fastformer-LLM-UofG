import pandas as pd
from tnlrv3.GraphFormers.src.models.tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from tnlrv3.GraphFormers.src.models.tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer
from tnlrv3.GraphFormers.src.models.tnlrv3.config import TuringNLRv3ForSeq2SeqConfig
from transformers import BertConfig
import torch
from torch.utils.data import Dataset, DataLoader
from claude_opus_fastformer import NewsRecommendationModel
from torch.optim import Adam
import time
from tqdm import tqdm

# Set the number of epochs
num_epochs = 5
MAX_USER_HISTORY = 50
MAX_IMPRESSION_NEWS = 50
dataset_type = "large"

# Set the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to MIND dataset files
NEWS_FILE_PATH = f'/mnt/c/Users/maxdo/Documents/University/data/{dataset_type}/train/news.tsv'
BEHAVIORS_FILE_PATH = f'/mnt/c/Users/maxdo/Documents/University/data/{dataset_type}/train/behaviors.tsv'

# Path to the UNILM model directory
model_path = "/mnt/c/Users/maxdo/Documents/University/data/unilm/minilm"

# Load the TuringNLRv3 model and tokenizer
config = TuringNLRv3ForSeq2SeqConfig.from_pretrained(model_path)
tokenizer = TuringNLRv3Tokenizer.from_pretrained(model_path)
model = TuringNLRv3ForSequenceClassification.from_pretrained(model_path, config=config)
model = model.to(device)

# Load and preprocess MIND dataset
news_df = pd.read_csv(NEWS_FILE_PATH, sep='\t', header=None, names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
behaviors_df = pd.read_csv(BEHAVIORS_FILE_PATH, sep='\t', header=None, names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

# Load the Fastformer model configuration
ff_config = BertConfig.from_json_file('fastformer_claude_opus.json')

# Function to preprocess text and generate embeddings
def generate_embeddings(texts, tokenizer, model, max_length=config.hidden_size, batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i:i+batch_size]
        preprocessed_texts = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        preprocessed_texts = preprocessed_texts.to(device)
        with torch.no_grad():
            outputs = model(**preprocessed_texts)
            batch_embeddings = outputs[0].mean(dim=1)
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings)

# Process news titles and generate embeddings
print("Generating news embeddings...")
start_time = time.time()
news_titles = news_df['Title'].tolist()
news_embeddings = generate_embeddings(news_titles, tokenizer, model)
end_time = time.time()
duration = end_time - start_time
print(f"News embeddings generated in {duration:.2f} seconds")

# Create a dictionary mapping news IDs to their embeddings
news_embedding_dict = dict(zip(news_df['NewsID'], news_embeddings))

class MINDDataset(Dataset):
    def __init__(self, behaviors_df, news_embedding_dict):
        self.behaviors_df = behaviors_df
        self.news_embedding_dict = news_embedding_dict

    def __len__(self):
        return len(self.behaviors_df)

    def __getitem__(self, idx):
        row = self.behaviors_df.iloc[idx]
        # print(row)
        
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
            if impression:
                news_id, label = impression.split('-')
                if news_id in self.news_embedding_dict:
                    impression_embds.append(self.news_embedding_dict[news_id])
                    labels.append(int(label))
        
        # Pad user history and impression news to MAX_USER_HISTORY and MAX_IMPRESSION_NEWS
        # print("user_history_embds shape:", len(user_history_embds))
        user_history_embds = user_history_embds[:MAX_USER_HISTORY]
        # print("user_history_embds shape:", len(user_history_embds))
        user_history_embds += [torch.zeros(ff_config.hidden_size)] * (MAX_USER_HISTORY - len(user_history_embds))
        # print("user_history_embds shape:", len(user_history_embds))
        impression_embds = impression_embds[:MAX_IMPRESSION_NEWS]
        impression_embds += [torch.zeros(ff_config.hidden_size)] * (MAX_IMPRESSION_NEWS - len(impression_embds))
        labels = labels[:MAX_IMPRESSION_NEWS]
        labels += [0] * (MAX_IMPRESSION_NEWS - len(labels))

        # for imp, embd in zip(impression_embds, user_history_embds):
        #     print(embd.shape, imp.shape)
        user_history_embds = torch.stack(user_history_embds)
        impression_embds = torch.stack(impression_embds)
        labels = torch.tensor(labels)

        return user_history_embds, impression_embds, labels

# Create DataLoader
train_dataset = MINDDataset(behaviors_df, news_embedding_dict)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the Fastformer model
fastformer_model = NewsRecommendationModel(ff_config, news_embedding_dim=ff_config.hidden_size, num_classes=MAX_IMPRESSION_NEWS, device=device)
fastformer_model = fastformer_model.to(device)

# Set up the optimizer
optimizer = Adam(fastformer_model.parameters(), lr=1e-5)

# Training loop
for epoch in range(num_epochs):
    fastformer_model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for user_history_embds, impression_embds, labels in progress_bar:
        user_history_embds = user_history_embds.to(device)
        impression_embds = impression_embds.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, scores = fastformer_model(user_history_embds, impression_embds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(loss=avg_loss)

    # Save the model after every epoch
    torch.save(fastformer_model.state_dict(), f"trained-models/fastformer_model_epoch_{epoch+1}.pth")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# Save the final model
torch.save(fastformer_model.state_dict(), "trained-models/fastformer_model_final.pth")