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
import torch.nn.functional as F

# Log the start time
with open('cnn_output.txt', 'w') as file:
    start_time = datetime.now()
    file.write(f"Start time: {start_time}\n")
    file.flush()

    # Load your data
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    embeddings_path = os.path.join(parent_dir, "2-5Seconds/Embeddings2-5.json")

    # Local Dataset
    # current_dir = os.getcwd()
    # parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    # embeddings_path = os.path.join(parent_dir, "Embeddings_5.5Seconds.json")

    file.write('Opening file to read\n')
    with open(embeddings_path, 'r') as f:
        data = json.load(f)

    file_names = [entry['fileName'] for entry in data]
    numpy_arrays = [np.array(entry['numpyArray']) for entry in data]
    tensor_arrays = [torch.tensor(arr).float() for arr in numpy_arrays]
    file.write('File has been read\n')
    file.flush()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = []
    tensor_arrays_balanced = []

    for i, file_name in enumerate(file_names): 
        if ('MC01' in file_name or 'FCO1' in file_name or 'MC04/Session2' in file_name):
            class_names.append('Healthy')
            tensor_arrays_balanced.append(tensor_arrays[i])
        elif ('F04' in file_name or 'M03' in file_name):
            class_names.append('VeryLow')
            tensor_arrays_balanced.append(tensor_arrays[i])
        elif ('F01' in file_name or 'M05' in file_name or 'F03/Session2' in file_name or 'F03/Session3' in file_name ):
            class_names.append('Low')
            tensor_arrays_balanced.append(tensor_arrays[i])
        elif ('M01' in file_name or 'M01' in file_name or 'M02' in file_name or 'M04' in file_name):
            class_names.append('Medium')
            tensor_arrays_balanced.append(tensor_arrays[i])

    tensor_arrays = tensor_arrays_balanced
    # tensor_arrays.to(device)
    unique_class_names = ['Healthy', 'Low', 'VeryLow', 'Medium']
    class_to_number = {class_name: i for i, class_name in enumerate(unique_class_names)}
    class_numbers = [class_to_number[name] for name in class_names]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        file.write("RUNNING ON GPU\n")
        file.flush()

    else:
        file.write("Running ON CPU\n")
        file.flush()

    class CustomEmbeddingDataset(Dataset):
        def __init__(self, embeddings, class_numbers):
            self.embeddings = embeddings
            self.class_numbers = class_numbers

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, index):
            batch_embeddings = self.embeddings[index]
            batch_class_numbers = self.class_numbers[index]
            tensor_batch_class_numbers = torch.tensor(batch_class_numbers)
            return batch_embeddings, tensor_batch_class_numbers

    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            # 1 since only black and white input of image
            # 6 is the number of outputs
            # 5 is the kernel size 
            self.conv1 = nn.Conv2d(1, 16, kernel_size = 3)
            # width = (375 - 3) / 1 + 1= 373
            # height = (1280 -3) / 1  + 1= 1278
            self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
            # width = roundDown((373 - 2) / 2) + 1 = 186
            # height = roundDown((1278 - 2 ) / 2) + 1 = 639
            self.conv2 = nn.Conv2d(16, 32, kernel_size = 3)
            # width = (186 - 3) / 1 + 1 = 184
            # height = (639 - 3) / 1  + 1 = 637

            # apply pool again 
            # width = roundDown((184 - 2) / 2)  + 1 = 92
            # height = roundDown(637 - 2) / 2 + 1) = 318

            self.conv3 = nn.Conv2d(32, 64, kernel_size = 5)
            # width = (92 - 5) / 1 + 1 = 88
            # height = (318 - 5) / 1 + 1 = 314

            # apply pool again
            # width = roundDown((88 - 2) / 2)  + 1  = 44
            # height = (314 - 2) / 2 + 1 = 157

            self.conv4 = nn.Conv2d(64, 128, kernel_size = 5)
            # width = roundDown(44 - 5) / 1)  + 1  = 40
            # height = (157 -5) / 1 + 1 = 153

            # apply pool again
            # width = roundDown((40 -2) / 2) + 1 = 20 
            # height = roundDown((153 - 2) / 2 ) + 1 = 76

            self.fc1 = nn.Linear(20 * 76 * 128, 4)

        def forward(self, x):
            # -> n, 3, 32, 32
            x = self.pool(F.relu(self.conv1(x)))  
            x = self.pool(F.relu(self.conv2(x)))  
            x = self.pool(F.relu(self.conv3(x)))  
            x = self.pool(F.relu(self.conv4(x)))  

            x = x.view(-1, 20 * 76 * 128)           
            # x = F.softmax(self.fc1(x))       
            x = self.fc1(x)                          
            return x

    input_size = 1280
    hidden_size = 128
    num_layers = 2
    num_classes = 4
    num_epochs = 100
    batch_size = 4
    initial_learning_rate = 0.01
    patience = 18
    k_folds = 10
    # patience = 2
    # k_folds = 2
    sequence_length = 375

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    dataset = CustomEmbeddingDataset(tensor_arrays, class_numbers)

    fold_results = []
    file.write("TRAINING THE MODEL\n")
    file.flush()

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        output_string = "FOLD NUMBER " + str(fold + 1) + "\n"
        file.write(output_string)
        file.flush()
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_validation_x = [x[0] for x in train_subset]
        train_validation_y = [x[1].item() for x in train_subset]
        x_test = [x[0] for x in test_subset]
        y_test = [x[1].item() for x in test_subset]
        x_train, x_validation, y_train, y_validation = train_test_split(train_validation_x, train_validation_y, test_size=0.1, random_state=42)

        train_data_set = CustomEmbeddingDataset(x_train, y_train)
        validation_data_set = CustomEmbeddingDataset(x_validation, y_validation)
        test_data_set = CustomEmbeddingDataset(x_test, y_test)
        
        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

        model = ConvNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=0.00001)

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images = images.unsqueeze(1).to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss = total_loss + loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.unsqueeze(1).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss = val_loss + loss.item()
            
            val_loss /= len(validation_loader)
            file.write(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']}\n")
            current_time = datetime.now()
            file.write(f"Start time: {current_time}\n")
            file.flush()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f'CNN_best_model_fold_{fold}.pth')
            elif epoch - best_epoch >= patience:
                file.write(f"Early stopping at epoch {epoch + 1}\n")
                file.flush()
                break

        model.load_state_dict(torch.load(f"CNN_best_model_fold_{fold}.pth"))

        # Evaluation
        model.eval()
        y_true, y_pred = [], []
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(num_classes)]
        n_class_samples = [0 for _ in range(num_classes)]
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.unsqueeze(1) 
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # max returns (value ,index)
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
                    file.write(f"Accuracy of {unique_class_names[i]}: {current_acc} %\n")
                    file.flush()
                else:
                    class_accs.append(0.0)
                    file.write(f"Accuracy of {unique_class_names[i]}: No samples\n")
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
        file.write(f"Average accuracy of {unique_class_names[i]}: {avg_class_accuracy[i]:.2f}%\n")
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
