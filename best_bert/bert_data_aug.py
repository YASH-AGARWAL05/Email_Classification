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
def extract_first_message(email_body):
    lines = email_body.split("\n")
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\s*(On .* wrote:|From:|Sent:|Subject:|To:|Cc:)', line, re.IGNORECASE):
            break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

# Function to remove stopwords and apply stemming
def preprocess_text(text):
    words = text.split()
    cleaned_words = [stemmer.lemmatize(word.lower()) for word in words]
    return " ".join(cleaned_words)

# Labels
labels = ["Confirmation", "Enquiry", "Complaint", "Others"]

# Initialize and fit the encoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Load datasets
df_train = pd.read_csv("gpt_1000.csv")
df_test = pd.read_csv("added_test.csv")

df_train = df_train[df_train["label"].isin(labels)]
df_test = df_test[df_test["label"].isin(labels)]

# Apply preprocessing:
# 1. Remove stopwords and apply stemming to both train & test
df_train["body"] = df_train["body"].astype(str).apply(preprocess_text)
df_test["body"] = df_test["body"].astype(str).apply(preprocess_text)
df_test.to_csv("cleaned_emails_real_dataset.csv")

# Convert labels to numerical values
df_train["label"] = df_train["label"].map(lambda x: label_encoder.transform([x])[0])
df_test["label"] = df_test["label"].map(lambda x: label_encoder.transform([x])[0])

# Take 10% of test data while preserving label distribution
test_for_train = pd.DataFrame()
remaining_test = pd.DataFrame()

# Get 10% from each label class
for label in df_test['label'].unique():
    label_subset = df_test[df_test['label'] == label]
    
    # Stratified split to get 10% for each label
    label_train, label_test = train_test_split(
        label_subset, 
        test_size=0.8,  # Keep 10% for training
        random_state=25
    )
    
    test_for_train = pd.concat([test_for_train, label_train], ignore_index=True)
    remaining_test = pd.concat([remaining_test, label_test], ignore_index=True)

# Verify the split (for debugging)
print(f"Original test set size: {len(df_test)}")
print(f"Test data for training: {len(test_for_train)} ({len(test_for_train)/len(df_test)*100:.1f}%)")
print(f"Remaining test data: {len(remaining_test)} ({len(remaining_test)/len(df_test)*100:.1f}%)")
print("Label distribution in test_for_train:")
for label in test_for_train['label'].unique():
    count = sum(test_for_train['label'] == label)
    orig_count = sum(df_test['label'] == label)
    print(f"Label {label_encoder.inverse_transform([label])[0]}: {count}/{orig_count} ({count/orig_count*100:.1f}%)")

# Combine with original training data
combined_train = pd.concat([df_train, test_for_train], ignore_index=True)

# Define BERT tokenizer
print("Setting up tokenizer...")
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

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
            "email": email,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Create a custom PyTorch module
class BertEmailClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=4, hidden_dropout_prob=0.1, 
                 attention_probs_dropout_prob=0.1, max_length=256):
        super(BertEmailClassifier, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_length = max_length
        
        config = BertConfig.from_pretrained(
            self.model_name,
            local_files_only=True,
            num_labels=self.num_labels,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob
        )
        
        self.bert = BertForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

# Create train, validation from the combined dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    combined_train["body"], combined_train["label"], test_size=0.2, random_state=55
)

print(collections.Counter(val_labels))

# Create datasets
train_dataset = EmailDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = EmailDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
test_dataset = EmailDataset(remaining_test["body"].tolist(), remaining_test["label"].tolist(), tokenizer)

# Define fixed hyperparameters (using the best params from the grid search as an example)
MODEL_PARAMS = {
    'learning_rate': 1e-5,
    'hidden_dropout_prob': 0.3,
    'attention_probs_dropout_prob': 0.3,
    'batch_size': 8,
    'weight_decay': 0.01,
    'epochs': 5
}

class EmailClassifierTrainer:
    def __init__(self, model_params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(label_encoder.classes_)
        self.model_params = model_params
        print(f"Using device: {self.device}")
        print(f"Training with parameters: {self.model_params}")
    
    def train_model(self, train_dataset, val_dataset, test_dataset):
        # Create data loaders with specified batch size
        train_loader = DataLoader(train_dataset, batch_size=self.model_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.model_params['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=self.model_params['batch_size'])
        
        # Initialize model with specified parameters
        model = BertEmailClassifier(
            num_labels=self.num_classes,
            hidden_dropout_prob=self.model_params['hidden_dropout_prob'],
            attention_probs_dropout_prob=self.model_params['attention_probs_dropout_prob']
        ).to(self.device)
        
        # Setup optimizer
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.model_params['learning_rate'], 
            weight_decay=self.model_params['weight_decay']
        )
        
        print(f"\nTraining model for {self.model_params['epochs']} epochs...")
        # Train the model and track metrics
        train_losses, val_losses, train_accs, val_accs = self.train_model_with_history(
            model, train_loader, val_loader, optimizer, 
            epochs=self.model_params['epochs']
        )
        
        # Evaluate on test set
        self.evaluate_model(model, test_loader, label_encoder)
        
        # Plot training curves
        self.plot_metrics(train_losses, val_losses, train_accs, val_accs)
        
        # Save model
        torch.save(model.state_dict(), "bert_email_classifier.pt")
        
        # Save parameters
        with open("model_params.pkl", "wb") as f:
            pickle.dump(self.model_params, f)
            
        return model
    
    def train_model_with_history(self, model, train_loader, val_loader, optimizer, epochs=3):
        """Full training function with metrics tracking"""
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        loss_fn = nn.CrossEntropyLoss()
        lr_scheduler = get_scheduler(
            "linear", 
            optimizer=optimizer, 
            num_warmup_steps=int(len(train_loader) * 0.1), 
            num_training_steps=len(train_loader) * epochs
        )
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                
                total_train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            train_losses.append(total_train_loss / len(train_loader))
            train_accs.append(correct_train / total_train)
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    total_val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            val_losses.append(total_val_loss / len(val_loader))
            val_accs.append(correct_val / total_val)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")
        
        return train_losses, val_losses, train_accs, val_accs
    
    def evaluate_model(self, model, test_loader, label_encoder):
        """Evaluate model on test set and save results"""
        model.eval()
        all_preds, all_labels, all_mails = [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                email = batch["email"]
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_mails.extend(email)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
        
        # Generate and print confusion matrix
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
        print("\nConfusion Matrix:")
        print(cm)
        
        # Create DataFrame for classified emails
        classified_df = pd.DataFrame({
            "Email": all_mails,
            "True Label": label_encoder.inverse_transform(all_labels),
            "Predicted Label": label_encoder.inverse_transform(all_preds)
        })
        
        # Filter correct and misclassified emails
        correct_df = classified_df[classified_df["True Label"] == classified_df["Predicted Label"]]
        misclassified_df = classified_df[classified_df["True Label"] != classified_df["Predicted Label"]]
        
        # Save to CSV
        correct_df.to_csv("correct_classified_emails.csv", index=False)
        misclassified_df.to_csv("misclassified_emails.csv", index=False)
        print("Correctly classified emails saved to 'correct_classified_emails.csv'")
        print("Misclassified emails saved to 'misclassified_emails.csv'")
    
    def plot_metrics(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training and validation metrics"""
        # Plot training and validation loss
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.savefig("model_loss.png")
        
        # Plot training and validation accuracy
        plt.figure(figsize=(10, 4))
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(val_accs, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training & Validation Accuracy")
        plt.legend()
        plt.savefig("model_accuracy.png")

# Run the code
if __name__ == "__main__":
    trainer = EmailClassifierTrainer(MODEL_PARAMS)
    model = trainer.train_model(train_dataset, val_dataset, test_dataset)
    
    print("\nTraining complete!")
    print(f"Model parameters: {MODEL_PARAMS}")
    print("Model saved to 'bert_email_classifier.pt'")