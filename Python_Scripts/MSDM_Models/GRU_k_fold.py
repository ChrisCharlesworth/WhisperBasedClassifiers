import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, jaccard_score, matthews_corrcoef, hamming_loss
from sklearn.preprocessing import LabelBinarizer
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from datetime import datetime


# Log the start time
with open('gru_output.txt', 'w') as file:
    start_time = datetime.now()
    file.write(f"Start time: {start_time}\n")
    file.flush()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        file.write("RUNNING ON GPU\n")
        file.flush()

    else:
        file.write("Running ON CPU\n")
        file.flush()

    class CustomEmbeddingDataset(Dataset):
        def __init__(self, embeddings_folder_path):
            self.embeddings_folder_path = embeddings_folder_path
            self.embeddings_json_files = [f for f in os.listdir(self.embeddings_folder_path) if f.endswith(".json")]

        def __len__(self):
            return len(self.embeddings_json_files)

        def __getitem__(self, index):
            file_path = os.path.join(self.embeddings_folder_path, self.embeddings_json_files[index])
            with open(file_path, 'r') as f:
                data = json.load(f)
            numpy_array = data["numpyArray"]
            tensor_arrays = torch.tensor(numpy_array)
            class_numbers = data["classNumber"] 
            return tensor_arrays, class_numbers

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(RNN, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.rnn(x, h0)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    input_size = 1280
    hidden_size = 128
    num_layers = 2
    num_classes = 4
    num_epochs = 100
    batch_size = 64
    initial_learning_rate = 0.01
    patience = 13
    k_folds = 10
    # patience = 3
    # k_folds = 2
    sequence_length = 40

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
    json_folder = os.path.join(parent_dir, "MSDM_Embeddings_2")
    dataset = CustomEmbeddingDataset(json_folder)

    fold_results = []
    file.write("TRAINING THE MODEL\n")
    file.flush()

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        output_string = "FOLD NUMBER " + str(fold + 1) + "\n"
        file.write(output_string)
        file.flush()
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_size = 0.9  # 90% for training
        validation_size = 0.1
        train_data_set, validation_data_set = torch.utils.data.random_split(train_subset, [train_size, validation_size])

        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=0.00001)

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images = images.reshape(-1, sequence_length, input_size).to(device)
                # images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.reshape(-1, sequence_length, input_size).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(validation_loader)
            file.write(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']}\n")
            current_time = datetime.now()
            file.write(f"Start time: {current_time}\n")
            file.flush()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f'GRU_best_model_fold_{fold}.pth')
            elif epoch - best_epoch >= patience:
                file.write(f"Early stopping at epoch {epoch + 1}\n")
                file.flush()
                break

        model.load_state_dict(torch.load(f"GRU_best_model_fold_{fold}.pth"))

        # Evaluation
        model.eval()
        y_true, y_pred = [], []
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(num_classes)]
        n_class_samples = [0 for _ in range(num_classes)]
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            file.write(f"Accuracy of the network: {acc} %\n")
            file.flush()

            class_accs = []
            for i in range(num_classes):
                if n_class_samples[i] != 0:
                    current_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                    class_accs.append(current_acc)
                    file.write(f"Accuracy of {i}: {current_acc} %\n")
                    file.flush()
                else:
                    class_accs.append(0.0)
                    file.write(f"Accuracy of {i}: No samples\n")
                    file.flush()

        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        jaccard_micro = jaccard_score(y_true, y_pred, average='micro')
        jaccard_macro = jaccard_score(y_true, y_pred, average='macro')
        jaccard_weighted = jaccard_score(y_true, y_pred, average='weighted')
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        y_pred_bin = lb.transform(y_pred)
        mccs = [matthews_corrcoef(y_true_bin[:, i], y_pred_bin[:, i]) for i in range(len(lb.classes_))]
        average_mcc = sum(mccs) / len(mccs)
        hl = hamming_loss(y_true_bin, y_pred_bin)

        file.write(f"Fold: {fold}\n")
        file.write(f"Accuracy: {acc:.2f}%\n")
        file.write(f"F1 Micro: {f1_micro:.4f}\n")
        file.write(f"F1 Macro: {f1_macro:.4f}\n")
        file.write(f"F1 Weighted: {f1_weighted:.4f}\n")
        file.write(f"Jaccard Micro: {jaccard_micro:.4f}\n")
        file.write(f"Jaccard Macro: {jaccard_macro:.4f}\n")
        file.write(f"Jaccard Weighted: {jaccard_weighted:.4f}\n")
        file.write(f"MCC: {average_mcc:.4f}\n")
        file.write(f"Hamming Loss: {hl:.4f}\n")
        file.flush()

        fold_results.append({
            'fold': fold,
            'accuracy': acc,
            'class_accuracy': class_accs,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'jaccard_micro': jaccard_micro,
            'jaccard_macro': jaccard_macro,
            'jaccard_weighted': jaccard_weighted,
            'mcc': average_mcc,
            'hamming_loss': hl
        })

    # Compute and print the average metrics
    avg_accuracy = sum(result['accuracy'] for result in fold_results) / k_folds
    avg_class_accuracy = [sum(result['class_accuracy'][i] for result in fold_results) / k_folds for i in range(num_classes)]
    avg_f1_micro = sum(result['f1_micro'] for result in fold_results) / k_folds
    avg_f1_macro = sum(result['f1_macro'] for result in fold_results) / k_folds
    avg_f1_weighted = sum(result['f1_weighted'] for result in fold_results) / k_folds
    avg_jaccard_micro = sum(result['jaccard_micro'] for result in fold_results) / k_folds
    avg_jaccard_macro = sum(result['jaccard_macro'] for result in fold_results) / k_folds
    avg_jaccard_weighted = sum(result['jaccard_weighted'] for result in fold_results) / k_folds
    avg_mcc = sum(result['mcc'] for result in fold_results) / k_folds
    avg_hamming_loss = sum(result['hamming_loss'] for result in fold_results) / k_folds

    file.write(f"Average accuracy: {avg_accuracy:.2f}%\n")
    file.flush()
    for i in range(num_classes):
        file.write(f"Average accuracy of {i}: {avg_class_accuracy[i]:.2f}%\n")
    file.write(f"Average F1 Score (Micro): {avg_f1_micro:.4f}\n")
    file.write(f"Average F1 Score (Macro): {avg_f1_macro:.4f}\n")
    file.write(f"Average F1 Score (Weighted): {avg_f1_weighted:.4f}\n")
    file.write(f"Average Jaccard Score (Micro): {avg_jaccard_micro:.4f}\n")
    file.write(f"Average Jaccard Score (Macro): {avg_jaccard_macro:.4f}\n")
    file.write(f"Average Jaccard Score (Weighted): {avg_jaccard_weighted:.4f}\n")
    file.write(f"Average MCC: {avg_mcc:.4f}\n")
    file.write(f"Average Hamming Loss: {avg_hamming_loss:.4f}\n")

    # Log the end time
    end_time = datetime.now()
    file.write(f"End time: {end_time}\n")

    # Optionally, you can also log the duration
    duration = end_time - start_time
    file.write(f"Duration: {duration}\n")
