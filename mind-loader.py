import pandas as pd
from tnlrv3.GraphFormers.src.models.tnlrv3.modeling import TuringNLRv3ForSequenceClassification, TuringNLRv3Model
from tnlrv3.GraphFormers.src.models.tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer
from tnlrv3.GraphFormers.src.models.tnlrv3.config import TuringNLRv3ForSeq2SeqConfig
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from keras.optimizers import Adam
import numpy as np
from fastformer import Fastformer
from tqdm import tqdm
import tensorflow as tf

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


# Process news titles and generate embeddings
news_titles = news_df['Title'].tolist()
news_embeddings = generate_embeddings(news_titles)

# Mapping from news ID to its index in news_embeddings
news_to_idx = {news_id: idx for idx, news_id in enumerate(news_df['NewsID'])}

# Create user profiles
user_profiles = create_user_profiles(behaviors_df, news_embeddings, news_to_idx)



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

# Model architecture
class NewsRecommendationModel(torch.nn.Module):
    def __init__(self, news_emb_dim, user_emb_dim):
        super(NewsRecommendationModel, self).__init__()
        news_emb_dim = news_embeddings.shape[1]
        user_emb_dim = next(iter(user_profiles.values())).shape[0]  # Assuming all user profiles have the same size

        # Define layers
        self.news_processor = Fastformer(news_emb_dim, 512)
        self.user_processor = Fastformer(user_emb_dim, 512)
        self.final_layer = torch.nn.Linear(512, 1)  # Output layer

    def forward(self, news_emb, user_emb):
        # Convert PyTorch tensors to NumPy arrays
        news_emb_np = news_emb.detach().cpu().numpy()
        user_emb_np = user_emb.detach().cpu().numpy()

        # Convert NumPy arrays to TensorFlow tensors
        news_emb_tf = tf.convert_to_tensor(news_emb_np)
        user_emb_tf = tf.convert_to_tensor(user_emb_np)

        # Ensure the input tensors are 3D
        if len(news_emb_tf.shape) == 2:
            news_emb_tf = tf.expand_dims(news_emb_tf, 1)
        if len(user_emb_tf.shape) == 2:
            user_emb_tf = tf.expand_dims(user_emb_tf, 1)

        # Process news and user embeddings
        news_features = self.news_processor(news_emb_tf)
        user_features = self.user_processor(user_emb_tf)
        combined_features = torch.cat([news_features, user_features], dim=1)
        
        # Final prediction
        output = self.final_layer(combined_features)
        return output

# Instantiate model
model = NewsRecommendationModel(news_embeddings, user_profiles)
# model.compile(optimizer=Adam(), loss='binary_crossentropy')
    
class NewsUserDataset(Dataset):
    def __init__(self, news_embeddings, user_profiles, labels):
        self.news_embeddings = news_embeddings
        self.user_profiles = user_profiles  # user_profiles is already a list
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
    # Create a mapping of user ID to clicked news IDs
    user_clicks = {}
    for _, row in behaviors_df.iterrows():
        clicked_news = row['History'].split() if isinstance(row['History'], str) else []
        user_clicks[row['UserID']] = set(clicked_news)

    # Generate labels
    labels = []
    for _, news_row in news_df.iterrows():
        news_id = news_row['NewsID']
        for user_id in user_clicks:
            labels.append(1 if news_id in user_clicks[user_id] else 0)

    return labels

labels = generate_labels(news_df, behaviors_df)




# Convert user_profiles to a list of tensors
user_embeddings = [torch.tensor(profile) for profile in user_profiles.values()]

# Prepare the dataset
dataset = NewsUserDataset(news_embeddings, user_embeddings, torch.tensor(labels))

# Initialize the DataLoader
trainloader = DataLoader(dataset, batch_size=32, shuffle=True)


num_epochs = 5
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
with tf.device('/GPU:0'):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}")
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Unpack the inputs
            news_emb, user_emb = inputs

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(news_emb, user_emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

print('Finished Training')



# save the model
torch.save(model, 'fastformer-demo')

