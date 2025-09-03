import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_scheduler

# Download stopwords
# nltk.download("stopwords")
# nltk.download('wordnet')
# nltk.download('omw-1.4')
stop_words = set(stopwords.words("english"))
stemmer = WordNetLemmatizer()

# Function to extract first message in email chain (for test data only)
def extract_first_message(text):
    # This function should be implemented based on your specific email format
    # For now, we'll just return the text as is
    return text

# Function to remove stopwords and apply stemming
def preprocess_text(text):
    words = text.split()
    cleaned_words = [stemmer.lemmatize(word.lower()) for word in words]
    return " ".join(cleaned_words)

# Load label encoder 5e-5
# with open("label_encoder_bert.pkl", "rb") as f:
#     label_encoder = pickle.load(f)

labels = ["Confirmation", "Enquiry","Complaint","Others"]

# Initialize and fit the encoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Load datasets
df_train = pd.read_csv("gpt_1700.csv")
df_test = pd.read_csv("added_test.csv")

df_train=df_train[df_train["label"].isin(labels)]
df_test=df_test[df_test["label"].isin(labels)]
# Apply preprocessing:
# 1. Remove stopwords and apply stemming to both train & test #
df_train["body"] = df_train["body"].astype(str).apply(preprocess_text)
df_test["body"] = df_test["body"].astype(str).apply(preprocess_text)
df_test.to_csv("cleaned_emails_real_dataset.csv")

# Convert labels to numerical values
df_train["label"] = df_train["label"].map(lambda x: label_encoder.transform([x])[0])
df_test["label"] = df_test["label"].map(lambda x: label_encoder.transform([x])[0])

# Define BERT tokenizer
print("tokenizer")
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME,local_files_only=True)

# Custom dataset class
class EmailDataset(Dataset):
    def __init__(self, emails, labels, tokenizer, max_len=256):
        self.emails = emails
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.emails)

    def __getitem__(self, idx):
        email = str(self.emails[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            email, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )

        return {
            "email":email,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Create train, validation, and test datasets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train["body"], df_train["label"], test_size=0.2, random_state=55
)

print(collections.Counter(val_labels))

train_dataset = EmailDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = EmailDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
test_dataset = EmailDataset(df_test["body"].tolist(), df_test["label"].tolist(), tokenizer)

# Create DataLoaders
BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define BERT model
num_classes = len(label_encoder.classes_)
config = BertConfig.from_pretrained(MODEL_NAME,
                                    local_files_only=True,
                                    num_labels=num_classes,
                                    hidden_dropout_prob=0.3,  # Increase dropout to prevent overfitting
                                    attention_probs_dropout_prob=0.3) 
print("model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME,config=config)
model.to(device)

# Define optimizer, loss, and scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()
num_training_steps = len(train_loader) * 5 
num_warmup_steps = int(0.1 * num_training_steps)
# lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)

# Training function
def train_model(model, train_loader, val_loader, epochs=5):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            # Gradient clipping line removed
            optimizer.step()
            

            total_train_loss += loss.item()
            correct_train += (logits.argmax(dim=1) == labels).sum().item()
            total_train += labels.size(0)

        train_losses.append(total_train_loss / len(train_loader))
        train_accs.append(correct_train / total_train)

        # Validation phase
        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                correct_val += (logits.argmax(dim=1) == labels).sum().item()
                total_val += labels.size(0)

        val_losses.append(total_val_loss / len(val_loader))
        val_accs.append(correct_val / total_val)

        print(f"Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

    return train_losses, val_losses, train_accs, val_accs

# Train the model
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader)

# Save the trained model
torch.save(model.state_dict(), "bert_email_classifier.pt")
print("Model saved as 'bert_email_classifier.pt'")

# Evaluate on test set
model.eval()
all_preds, all_labels, all_mails = [], [], []

with torch.no_grad():
    for batch in test_loader:
        email, input_ids, attention_mask, labels = batch["email"], batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_mails.extend(email)

# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

with open("classification_report.txt","w") as f:
    f.write(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Create confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# Save loss curve with better filename
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.savefig("bert_loss_curve.png")

# Save accuracy curve with better filename
plt.figure(figsize=(10, 4))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.savefig("bert_accuracy_curve.png")

# Create DataFrame for misclassified emails
misclassified_df = pd.DataFrame({
    "Email": all_mails,
    "True Label": label_encoder.inverse_transform(all_labels),
    "Predicted Label": label_encoder.inverse_transform(all_preds)
})

# Save to CSV
misclassified_df.to_csv("email_class.csv", index=False)
print("Classification results saved to 'email_class.csv'")