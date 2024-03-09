import pandas as pd
from tnlrv3.GraphFormers.src.models.tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from tnlrv3.GraphFormers.src.models.tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer
from tnlrv3.GraphFormers.src.models.tnlrv3.config import TuringNLRv3ForSeq2SeqConfig
import torch
from torch.utils.data import Dataset, DataLoader
from fastformer_claude_opus import NewsRecommendationModel
from torch.optim import Adam

import time
from tqdm import tqdm

# Set the number of epochs
num_epochs = 5

# Set the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to MIND dataset files
NEWS_FILE_PATH = '/mnt/c/Users/maxdo/Documents/University/data/demo/train/news.tsv'
BEHAVIORS_FILE_PATH = '/mnt/c/Users/maxdo/Documents/University/data/demo/train/behaviors.tsv'

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

# Function to preprocess text and generate embeddings
def generate_embeddings(texts, tokenizer, model, max_length=512, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        preprocessed_texts = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        preprocessed_texts = preprocessed_texts.to(device)
        with torch.no_grad():
            outputs = model(**preprocessed_texts)
            batch_embeddings = outputs[0].mean(dim=1)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Process news titles and generate embeddings
news_titles = news_df['Title'].tolist()
news_embeddings = generate_embeddings(news_titles, tokenizer, model)

# Mapping from news ID to its index in news_embeddings
news_to_idx = {news_id: idx for idx, news_id in enumerate(news_df['NewsID'])}

# Function to create user profiles
def create_user_profiles(behaviors_df, news_embeddings, news_to_idx):
    user_profiles = {}
    for _, row in behaviors_df.iterrows():
        user_id = row['UserID']
        clicked_news = row['History'].split() if isinstance(row['History'], str) else []
        clicked_embeddings = [news_embeddings[news_to_idx[news_id]] for news_id in clicked_news if news_id in news_to_idx]
        
        if clicked_embeddings:
            clicked_embeddings = torch.stack(clicked_embeddings)
            user_profile = torch.mean(clicked_embeddings, dim=0)  # Average pooling
            user_profiles[user_id] = user_profile.cpu().numpy()  # Convert back to numpy array for consistency
    return user_profiles

# Create user profiles
user_profiles = create_user_profiles(behaviors_df, news_embeddings, news_to_idx)

# Dataset preparation
class NewsRecommendationDataset(Dataset):
    def __init__(self, news_embeddings, user_profiles, labels):
        self.news_embeddings = news_embeddings
        self.user_profiles = user_profiles
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        num_users = len(self.user_profiles)
        news_idx = idx // num_users
        user_idx = idx % num_users

        news_emb = self.news_embeddings[news_idx]
        user_emb = self.user_profiles[user_idx]
        label = self.labels[idx]

        return (news_emb, user_emb), label

# Generate labels
def generate_labels(news_df, behaviors_df):
    user_clicks = {row['UserID']: set(row['History'].split()) for _, row in behaviors_df.iterrows() if isinstance(row['History'], str)}
    labels = []
    for news_id in news_df['NewsID']:
        for user_id, clicked_news in user_clicks.items():
            labels.append(1 if news_id in clicked_news else 0)
    return labels

labels = generate_labels(news_df, behaviors_df)

# Custom collate function for padding
def custom_collate_fn(batch):
    news_embeddings = []
    user_embeddings = []
    labels = []

    for (news_emb, user_emb), label in batch:
        news_embeddings.append(news_emb)
        user_embeddings.append(user_emb)
        labels.append(label)

    # Pad news embeddings
    padded_news_embeddings = torch.nn.utils.rnn.pad_sequence(news_embeddings, batch_first=True)

    # Stack user embeddings and labels
    user_embeddings = torch.stack(user_embeddings)
    labels = torch.tensor(labels)

    return (padded_news_embeddings, user_embeddings), labels

# Convert user_profiles to a list of tensors
user_embeddings = [torch.tensor(profile) for profile in user_profiles.values()]

# Prepare the dataset
dataset = NewsRecommendationDataset(news_embeddings, user_embeddings, torch.tensor(labels))

# Instantiate the Fastformer model
# Instantiate the Fastformer model
config.max_position_embeddings = 512  # Set the maximum position embeddings
model = NewsRecommendationModel(user_embedding_dim=news_embeddings.shape[1], num_classes=2, device=device)
model = model.to(device)

# Initialize the DataLoader
trainloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)


# Set up optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    start_time = time.time()
    
    running_loss = 0.0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}", unit="batch")
    
    model.train()
    for batch in progress_bar:
        (news_input, user_input), targets = batch
        news_input = news_input.to(device)
        user_input = user_input.to(device)
        targets = targets.to(device).long()  # Convert targets to long tensor
        
        optimizer.zero_grad()
        loss, logits = model(news_input, user_input, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        progress_bar.set_postfix({"Loss": running_loss / (progress_bar.n + 1)})
    
    epoch_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch {epoch+1} Duration: {epoch_duration:.2f} seconds")
    
    # Save the model after each epoch
    model_save_path = f"models/fastformer_model_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")

print("Training completed!")