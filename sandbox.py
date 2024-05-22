from models import SimpleModel, DeepModel, ConvModelShallow, AudioClassifier
from train import train
from dataset import MLPC_Dataset
import torch
from transformers import AutoModelForAudioClassification

dataset = MLPC_Dataset("mel")
train_loader, test_loader = dataset.get_data_loaders(batch_size=64, split=(0.80, 0.20))

model = AudioClassifier()

train(network=model, test_loader=test_loader, train_loader=train_loader,num_epochs=100, plot_loss=True, save_model=True, plot_accuracy=True)
