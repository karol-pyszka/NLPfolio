
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

from SentimentClassifier import SentimentClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8128
test_size = 10


def get_data_and_make_preprocess():

    # load pretrained model to create embeddings
    model = SentenceTransformer('bert-base-nli-mean-tokens').to(device)

    username_pattern = r'@[^\s]+'
    link_pattern = r'http\S+'

    data = pd.read_csv("data/training.1600000.processed.noemoticon.csv", header=None,
                       names=['target', 'id', 'date', 'flag', 'user', 'text'], encoding='latin-1')
    data = data[['text', 'target']]
    data['target'] = data['target'].map({0: 0, 2: 1, 4: 2})

    data = data.sample(n=BATCH_SIZE * test_size, random_state=42)

    # delete usernames and links
    data['text'] = data['text'].str.replace(username_pattern, 'username', regex=True)
    data['text'] = data['text'].str.replace(link_pattern, 'link', regex=True)
    data['text'] = data['text'].apply(preprocess_text)
    embeddings = []

    # Loop over batches of sentences to create sentence embeddings
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        # Get batch of sentences
        batch = data['text'][i:i + BATCH_SIZE].tolist()

        # Encode batch of sentences
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False).to(device)
        for emb in batch_embeddings:
            # Append each row to list
            embeddings.append(emb.detach().cpu().numpy())
    data['embeddings'] = embeddings
    data.to_csv('data_with_embeddings.csv', index=False)
    print(data.head(5))
    return data


def preprocess_text(text):
    text = text.lower()
    return text


def train_test_split_data(data):
    return train_test_split(data, test_size=0.2, random_state=42)


def main():

    # uncomment if you don't have file data_with_embeddings.csv
    # data = get_data_and_make_preprocess()
    # if upper is uncomment, comment this one
    data = pd.read_csv("data_with_embeddings.csv")

    data['embeddings'] = data['embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

    data['target'] = data['target'].astype(float)

    train_data, val_data = train_test_split_data(data)

    train_inputs = torch.tensor(np.array(train_data['embeddings'].tolist())).float().to(device)
    train_labels = torch.tensor((np.array(train_data['target'].tolist()))).long().to(device)
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_inputs = torch.tensor((np.array(val_data['embeddings'].tolist()))).float().to(device)
    val_labels = torch.tensor((np.array(val_data['target'].tolist()))).long().to(device)
    val_dataset = TensorDataset(val_inputs, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    input_dim = train_inputs.shape[-1]
    output_dim = 3
    hidden_dim = 128
    modelL = SentimentClassifier(input_dim, hidden_dim, output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelL.parameters())

    num_epochs = 30

    for epoch in tqdm(range(num_epochs)):
        loss_sum = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = modelL(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = loss_sum / len(train_loader)
        print("Epoch: " + str(epoch) + " Avg_loss: " + str(avg_loss))

    evaluate(modelL, val_loader, criterion)


def evaluate(modelL, dataloader, criterion):
    modelL.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = modelL(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_predictions)
    print("Avg_loss: " + str(avg_loss))
    print(confusion_matrix(all_labels, all_predictions))
    print(classification_report(all_labels, all_predictions))
    print("Acc: " + str(acc))
    return avg_loss, acc


if __name__ == "__main__":
    main()
