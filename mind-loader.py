import pandas as pd
from tnlrv3.GraphFormers.src.models.tnlrv3.modeling import TuringNLRv3ForSequenceClassification, TuringNLRv3Model
from tnlrv3.GraphFormers.src.models.tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer
from tnlrv3.GraphFormers.src.models.tnlrv3.config import TuringNLRv3ForSeq2SeqConfig
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from keras.optimizers import Adam
import numpy as np
from fastformer import NewsRecommendationModel
from tqdm import tqdm
import tensorflow as tf
from torch.nn import CrossEntropyLoss

# Set the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device
print("Device:", device)

# Paths to MIND dataset files
NEWS_FILE_PATH = '/mnt/c/Users/maxdo/Documents/University/data/demo/train/news.tsv'
BEHAVIORS_FILE_PATH = '/mnt/c/Users/maxdo/Documents/University/data/demo/train/behaviors.tsv'

# Path to the UniLM model directory
model_path = "/mnt/c/Users/maxdo/Documents/University/data/unilm/minilm"

# Load the TuringNLRv3 model and tokenizer
config =  TuringNLRv3ForSeq2SeqConfig.from_pretrained(model_path)
tokenizer = TuringNLRv3Tokenizer.from_pretrained(model_path)
model = TuringNLRv3ForSequenceClassification.from_pretrained(model_path, config=config)
# model = TuringNLRv3Model.from_pretrained(model_path, config=config)
model = model.to(device)


# Load and preprocess MIND dataset
news_df = pd.read_csv(NEWS_FILE_PATH, sep='\t', header=None, names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
behaviors_df = pd.read_csv(BEHAVIORS_FILE_PATH, sep='\t', header=None, names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

# Function to preprocess text and generate embeddings
def generate_embeddings(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        preprocessed_texts = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        preprocessed_texts = preprocessed_texts.to(device)
        with torch.no_grad():
            outputs = model(**preprocessed_texts)
            # print("Shape of outputs:", outputs[0].shape)  # Add this line
            batch_embeddings = outputs[0].mean(dim=1)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Function to create user profiles
def create_user_profiles(behaviors_df, news_embeddings, news_to_idx):
    news_embeddings = torch.tensor(news_embeddings).to(device)
    user_profiles = {}
    for _, row in behaviors_df.iterrows():
        user_id = row['UserID']
        clicked_news = row['History'].split() if isinstance(row['History'], str) else []
        clicked_embeddings = [news_embeddings[news_to_idx[news_id]] for news_id in clicked_news if news_id in news_to_idx]
        
        if clicked_embeddings:
            clicked_embeddings = torch.stack(clicked_embeddings)
            user_profile = torch.mean(clicked_embeddings, axis=0)  # Average pooling
            user_profiles[user_id] = user_profile.cpu().numpy()  # Convert back to numpy array for consistency
    return user_profiles

def prepare_user_interaction_data(behaviors_df, news_to_idx):
    """
    Convert user interactions into a sequence of news article indices.
    Assume behaviors_df has a column 'History' with space-separated news IDs that the user clicked on.
    """
    user_interactions = []

    # Maximum history length to consider
    max_history_len = 50

    for history_str in behaviors_df['History'].fillna(''):
        # Split the history string into news IDs
        history_ids = history_str.split()
        
        # Convert news IDs to indices based on news_to_idx mapping
        history_idxs = [news_to_idx[news_id] for news_id in history_ids if news_id in news_to_idx]
        
        # Pad or truncate the history to max_history_len
        padded_history = history_idxs[:max_history_len] + [0] * (max_history_len - len(history_idxs))
        
        user_interactions.append(padded_history)
    
    return torch.tensor(user_interactions)


# Process news titles and generate embeddings
news_titles = news_df['Title'].tolist()
news_embeddings = generate_embeddings(news_titles)

# Mapping from news ID to its index in news_embeddings
news_to_idx = {news_id: idx for idx, news_id in enumerate(news_df['NewsID'])}

# Create user profiles
user_profiles = create_user_profiles(behaviors_df, news_embeddings, news_to_idx)

# Assuming news_to_idx is a dictionary mapping news IDs to their corresponding indices
user_interactions_tensor = prepare_user_interaction_data(behaviors_df, news_to_idx)



# Dataset preparation (customize based on your data structure)
class NewsRecommendationDataset(Dataset):
    def __init__(self, news_embeddings, user_data, labels):
        self.news_embeddings = news_embeddings
        self.user_data = user_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        news_idx = idx % len(self.news_embeddings)  # Wrap around the indices
        return self.news_embeddings[news_idx], self.user_data[idx], self.labels[idx]



# Instantiate model
model = NewsRecommendationModel(news_embeddings, user_profiles, len(user_profiles))
    
class NewsUserDataset(Dataset):
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





#TODO finish genereting DataLoader
    
def generate_labels(news_df, behaviors_df):
    user_clicks = {row['UserID']: set(row['History'].split()) for _, row in behaviors_df.iterrows() if isinstance(row['History'], str)}
    labels = []
    for news_id in news_df['NewsID']:
        for user_id, clicked_news in user_clicks.items():
            labels.append(1 if news_id in clicked_news else 0)
    return labels

labels = generate_labels(news_df, behaviors_df)




# Convert user_profiles to a list of tensors
user_embeddings = [torch.tensor(profile) for profile in user_profiles.values()]

# Prepare the dataset
dataset = NewsUserDataset(news_embeddings, user_embeddings, torch.tensor(labels))

# Initialize the DataLoader
trainloader = DataLoader(dataset, batch_size=32, shuffle=True)



# debugging here:
num_unique_news = len(news_df['NewsID'].unique())
print("Number of unique news articles:", num_unique_news)
# Length of news_embeddings
print("Length of news_embeddings:", len(news_embeddings))
# Assuming num_users is correctly calculated
num_users = len(user_profiles)
print("Number of users:", num_users)

# Maximum index for news that should be accessed
max_news_idx = (len(labels) // num_users) - 1
print("Maximum index for news:", max_news_idx)
# Check if max_news_idx exceeds the length of news_embeddings
if max_news_idx >= len(news_embeddings):
    print("Error: Index exceeds length of news_embeddings.")
else:
    print("Index calculation is correct.")

# Expected total number of pairs
expected_pairs = num_unique_news * num_users
print("Expected total number of (news, user) pairs:", expected_pairs)
# Actual number of labels
print("Actual number of labels:", len(labels))
# Check if they match
if len(labels) != expected_pairs:
    print("Error: Number of labels does not match the expected number of pairs.")
else:
    print("Label count is correct.")

# # Test the Fastformer layer
# fastformer_test = Fastformer(news_embeddings.shape, 512)  # Use appropriate dimensions
# dummy_input = torch.randn(32, 1, news_embeddings.shape)  # Create a dummy input tensor of the correct shape
# try:
#     output = fastformer_test(dummy_input)
#     print("Output shape from Fastformer:", output.shape)
# except Exception as e:
#     print("Error in Fastformer:", e)


# num_epochs = 5
# criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters())

# # Train the model
# with tf.device('/GPU:0'):
#     for epoch in range(num_epochs):  # loop over the dataset multiple times
#         print(f"Epoch {epoch+1}")
#         running_loss = 0.0
#         for i, data in tqdm(enumerate(trainloader, 0)):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data

#             # Unpack the inputs
#             news_emb, user_emb = inputs

#             print("Batch news_emb shape:", news_emb.shape)  # Add this line
#             print("Batch user_emb shape:", user_emb.shape)  # Add this line

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = model(news_emb, user_emb)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                     (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
    
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = CrossEntropyLoss()

# Example training step
model.train()
for batch in trainloader:
    news_input_ids = batch['news_input_ids'].to(device)
    news_attention_mask = batch['news_attention_mask'].to(device)
    user_interaction_idxs = batch['user_interaction_idxs'].to(device)
    targets = batch['targets'].to(device)
    
    optimizer.zero_grad()
    logits = model(news_input_ids, news_attention_mask, user_interaction_idxs)
    
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()


print('Finished Training')



# save the model
torch.save(model, 'fastformer-demo')

