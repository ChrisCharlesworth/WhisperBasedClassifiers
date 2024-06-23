import os
from transformers import WhisperForConditionalGeneration, AutoProcessor
import torch
import librosa
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np
import wave

# Get the current directory
current_dir = os.getcwd()
# Navigate to the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Define the path to the data folder
data_folder = os.path.join(parent_dir, "data")

def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

def find_wav_files(directory):
    wav_files = []
    # Iterate over all items (files and directories) in the current directory
    for item in os.listdir(directory):
        # Construct the full path
        item_path = os.path.join(directory, item)
        # If it's a file and ends with .wav, and there's no 'C' in the path, and duration is longer than 2 seconds, add it to the list
        if os.path.isfile(item_path) and item.lower().endswith(".wav"):
            if get_wav_duration(item_path) > 2.5:
                file = item_path
                if ('MC01' in file or 'FCO1' in file or 'MC04/Session2' in file or 'F04' in file or 
                'M03' in file or 'F01' in file or 'M05' in file or 'F03/Session2' in file or 
                'F03/Session3' in file or 'M01' in file or 'M02' in file or 'M04' in file):
                    wav_files.append(item_path)
        # If it's a directory, recursively call the function
        elif os.path.isdir(item_path):
            wav_files.extend(find_wav_files(item_path))
    return wav_files

wav_files = find_wav_files(data_folder)
print(len(wav_files))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# directory = Path.cwd().parent.as_posix() + '/data/small_dataset'

audios = []
file_names = []
print('reading audio')
output_file_name = "Embeddings2-5.json"
with open(output_file_name, 'w') as json_file:
    json_file.write("[")  # Start of the JSON array
    # Iterate over files in the directory
    file_counter = 0
    for file_path in wav_files:
        if file_path.endswith(".wav"):
            audio, sr = librosa.load(file_path, sr=None)
            # audios.append(audio)
            input = processor(audio, return_tensors="pt", return_attention_mask=True, sampling_rate=16000)
            file_names.append(file_path)
            output = model.model.encoder(**input.to(dtype=torch.bfloat16), output_hidden_states=True)
            your_tensor = output.last_hidden_state[0]

            # Convert the tensor to a supported data type (e.g., float32) before converting to NumPy array
            your_tensor = your_tensor.float()  # Convert to float32
            numpy_array = your_tensor.detach().numpy()

            # ONLY TAKE FIRST 375 ie. first 7.5 seconds
            numpy_array = numpy_array[:375, :]

            if file_counter > 0:   
               json_file.write(", ")  # Add comma before each element except the first one
            json.dump({"fileName": file_path, "numpyArray": numpy_array.tolist()}, json_file) 
            print('Written to json for file ' + file_path + ' which is file number ' +  str(file_counter))
            file_counter += 1
    json_file.write("]")  # End of the JSON array
