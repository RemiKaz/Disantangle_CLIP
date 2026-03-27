import clip
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import argparse
import data
import models
import time

# Assuming MonumaiDataset and CLIPLinear are already defined as provided

def train_and_test_clip_linear(dataset,cbm, device='cuda',disantangled_version=False,restrict_samples=None,part=None,lr=0.01,
                               root_monumai=None,root_imagenet=None,root_cub=None,root_parts=None,
                               pth_metadata='metadata/imagnet_classes.json',pth_metadata_parts='metadata/file_parts_mapping.json'):
    # Initialize datasets for each phase

    if dataset == 'monumai':
        monumai_dataset_train = data.MonumaiDataset(root=root_monumai, phase='train', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                   reduce_samples=False,train_cbm_mode=True)
        monumai_dataset_val = data.MonumaiDataset(root=root_monumai, phase='val', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                    reduce_samples=False,train_cbm_mode=True)
        monumai_dataset_test = data.MonumaiDataset(root=root_monumai, phase='test', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                    reduce_samples=False,train_cbm_mode=True)

        # DataLoaders for training, validation, and testing
        train_loader = DataLoader(monumai_dataset_train, batch_size=32, shuffle=True)
        val_loader = DataLoader(monumai_dataset_val, batch_size=32, shuffle=False)
        test_loader = DataLoader(monumai_dataset_test, batch_size=32, shuffle=False)

        # Define concepts and classes relevant to your dataset
        list_concepts = [
            'horseshoe-arch', 'lobed-arch', 'pointed-arch', 'ogee-arch', 'trefoil-arch',
            'serliana', 'solomonic-column', 'pinnacle-gothic', 'porthole', 'broken-pediment',
            'rounded-arch', 'flat-arch', 'segmental-pediment', 'triangular-pediment',
            'lintelled-doorway'
        ]
        list_classes = ['Baroque', 'Gothic', 'Hispanic-Muslim', 'Renaissance']

    elif dataset == 'imagenet':

        imagenet_dataset_train = data.ImageNetCLIPDataset(root_imagenet, split='train', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                 device='cuda',pth_metadata=pth_metadata, pth_metadata_parts=pth_metadata_parts,
                 disantangled_version=disantangled_version,restrict_samples=restrict_samples,part=part)

        imagenet_dataset_val = data.ImageNetCLIPDataset(root_imagenet, split='val', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                 device='cuda',pth_metadata=pth_metadata, pth_metadata_parts=pth_metadata_parts,
                 disantangled_version=disantangled_version,restrict_samples=restrict_samples)

        imagenet_dataset_test = imagenet_dataset_val
        
        list_concepts = []
        for class_name in imagenet_dataset_train.metadata_parts.keys():
            list_concepts += imagenet_dataset_train.metadata_parts[class_name]

        list_classes = imagenet_dataset_train.metadata_imagnet.items()

        # DataLoaders for training, validation, and testing
        train_loader = DataLoader(imagenet_dataset_train, batch_size=4096, shuffle=True, num_workers=16, pin_memory=True)
        val_loader = DataLoader(imagenet_dataset_val, batch_size=4096, shuffle=False, num_workers=16, pin_memory=True)
        test_loader = DataLoader(imagenet_dataset_test, batch_size=4096, shuffle=False, num_workers=16, pin_memory=True)

    elif dataset == 'cub':
        cub_dataset_train = data.CUBDataset(root=root_cub, phase='train', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                   reduce_samples=False,device='cuda',clip_concepts=None,train_cbm_mode=True,
                                   disantangled_version=disantangled_version,class_restrict=None)
        cub_dataset_val = data.CUBDataset(root=root_cub, phase='val', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                   reduce_samples=False,device='cuda',clip_concepts=None,train_cbm_mode=True,
                                   disantangled_version=disantangled_version,class_restrict=None)
        cub_dataset_test = cub_dataset_val

        # DataLoaders for training, validation, and testing
        train_loader = DataLoader(cub_dataset_train, batch_size=32, shuffle=True)
        val_loader = DataLoader(cub_dataset_val, batch_size=32, shuffle=False)
        test_loader = DataLoader(cub_dataset_test, batch_size=32, shuffle=False)

        # Define concepts and classes relevant to your dataset
        list_concepts = cub_dataset_train.list_concepts
        list_classes = cub_dataset_train.list_classes

    elif dataset == 'cub_precompute':
        cub_dataset_train = data.CUBPrecomputeDataset(root_cub,
                                                      split='train', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                   device='cuda',disantangled_version=disantangled_version,restrict_samples=restrict_samples)
        cub_dataset_val = data.CUBPrecomputeDataset(root_cub,
                                                    split='val', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                   device='cuda',disantangled_version=disantangled_version,restrict_samples=restrict_samples)
        cub_dataset_test = cub_dataset_val

        # DataLoaders for training, validation, and testing
        train_loader = DataLoader(cub_dataset_train, batch_size=16, shuffle=True)
        val_loader = DataLoader(cub_dataset_val, batch_size=16, shuffle=False)
        test_loader = DataLoader(cub_dataset_test, batch_size=16, shuffle=False)

        # Define concepts and classes relevant to your dataset
        list_concepts = cub_dataset_train.list_concepts
        list_classes = cub_dataset_train.list_classes

    elif dataset == 'monumai_precompute':
        monumai_dataset_train = data.MonumaiPrecomputeDataset(root_monumai,
                                                      split='train', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                   device='cuda',disantangled_version=disantangled_version,restrict_samples=restrict_samples)
        monumai_dataset_val = data.MonumaiPrecomputeDataset(root_monumai,
                                                    split='val', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                   device='cuda',disantangled_version=disantangled_version,restrict_samples=restrict_samples)
        monumai_dataset_test = data.MonumaiPrecomputeDataset(root_monumai,
                                                    split='test', model_name='ViT-B-16', pretrained='laion2b_s34b_b88k',
                                   device='cuda',disantangled_version=disantangled_version,restrict_samples=restrict_samples)

        # DataLoaders for training, validation, and testing
        train_loader = DataLoader(monumai_dataset_train, batch_size=32, shuffle=True)
        val_loader = DataLoader(monumai_dataset_val, batch_size=32, shuffle=False)
        test_loader = DataLoader(monumai_dataset_test, batch_size=32, shuffle=False)

        # Define concepts and classes relevant to your dataset
        list_concepts = monumai_dataset_train.list_concepts
        list_classes = monumai_dataset_train.list_classes

    else:
        raise ValueError("Unsupported dataset. Use 'monumai'.")

    if cbm == 'clip_linear':
        # Initialize the CLIPLinear classifier
        classifier = models.CLIPLinear(list_concepts, list_classes, device=device,disantangled_version=disantangled_version)
    
    elif cbm == 'clip_linear_precompute' or cbm == 'clip_linear_precomputed':    
        # Initialize the CLIPLinear classifier
        classifier = models.CLIPLinearPrecomputed(list_concepts, list_classes,device=device)
    
    else:
        raise ValueError("Unsupported CBM type. Use 'clip_linear'.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(classifier.linear.parameters(), lr=lr, weight_decay=0.0)
    
    # Training loop
    num_epochs = 300

    best_val_accuracy = 0.0
    best_test_accuracy = 0.0

    for epoch in range(num_epochs):
        if 'precompute' in cbm:
            classifier.train()
            train_loss = 0.0
            flag_see = False
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                if flag_see:
                    print(batch['clip_scores'],batch['clip_scores'].shape)
                    flag_see = False
                image_embedings = batch['clip_scores'].to(device)
                labels = torch.tensor(batch['label_number'], dtype=torch.long).to(device)
                outputs = classifier(image_embedings)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.4f}')

            classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            flag_see = False
            with torch.no_grad():
                for batch in val_loader:
                    image_tensors = batch['clip_scores'].to(device)
                    labels = torch.tensor(batch['label_number'], dtype=torch.long).to(device)
                    outputs = classifier(image_tensors)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    if flag_see:
                        print(predicted,labels)
                        flag_see = False
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            print(f'Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

            test_accuracy = test_model_sim(classifier, test_loader, device)
            print(f'Test Accuracy: {test_accuracy:.4f}')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_test_accuracy = test_accuracy
                torch.save(classifier.state_dict(), f'best_model_{args.dataset}_{args.cbm}_{args.disantangled_version}.pt')

            # Train accuracy
            train_loss = 0.0
            correct = 0
            total = 0
            flag_see = False
            with torch.no_grad():
                for batch in train_loader:
                    image_tensors = batch['clip_scores'].to(device)
                    labels = torch.tensor(batch['label_number'], dtype=torch.long).to(device)
                    
                    outputs = classifier(image_tensors)
                    loss = criterion(outputs, labels)
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    if flag_see:
                        print(predicted,labels)
                        flag_see = False
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct / total
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
            
            test_accuracy = test_model_sim(classifier, train_loader, device)
            print(f'Train Accuracy: {test_accuracy:.4f}')

        else:
            classifier.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                image_tensors = batch['image_tensor'].to(device)
                labels = torch.tensor(batch['label_number'], dtype=torch.long).to(device)
                outputs = classifier(image_tensors)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.4f}')

            classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    image_tensors = batch['image_tensor'].to(device)
                    labels = torch.tensor(batch['label_number'], dtype=torch.long).to(device)
                    outputs = classifier(image_tensors)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            print(f'Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

            test_accuracy = test_model(classifier, test_loader, device)
            print(f'Test Accuracy: {test_accuracy:.4f}')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_test_accuracy = test_accuracy
                # save model
                torch.save(classifier.state_dict(), f'best_model_{args.dataset}_{args.cbm}_{args.disantangled_version}.pt')

    print(f'Best Val Accuracy: {best_val_accuracy:.4f}')
    print(f'Best Test Accuracy (at best val): {best_test_accuracy:.4f}')

def test_model(classifier, data_loader, device='cuda'):
    """Test the model on the provided data loader."""
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            # Extract image tensors and labels from batch
            image_tensors = batch['image_tensor'].to(device)
            labels = torch.tensor(batch['label_number'], dtype=torch.long).to(device)

            outputs = classifier(image_tensors)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def test_model_sim(classifier, data_loader, device='cuda'):
    """Test the model on the provided data loader."""
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            # Extract image tensors and labels from batch
            image_tensors = batch['clip_scores'].to(device)
            labels = torch.tensor(batch['label_number'], dtype=torch.long).to(device)

            outputs = classifier(image_tensors)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

if __name__ == '__main__':
    # Parser for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet', help='Path to the dataset root in [monumai, imagenet,cub]')
    parser.add_argument('--cbm', type=str, default='clip_linear_precompute', help='Type of concept bottleneck model to use')
    parser.add_argument('--disantangled_version', type=str, default='median-loc-random', help='Disantangled version to use')
    parser.add_argument('--restrict_samples', type=int, default=None, help='Number of samples to restrict to')
    parser.add_argument('--part', type=int, default=None, help='for precomputing in imagenet')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--root_monumai', type=str, default='/lustre/fswork/projects/rech/wqn/ufb58bn/DataRemi/OD-MonuMAI/MonuMAI_dataset', help='Root path to MonuMAI dataset')
    parser.add_argument('--root_imagenet', type=str, default='/lustre/fsmisc/dataset/imagenet', help='Root path to ImageNet dataset')
    parser.add_argument('--root_cub', type=str, default='/lustre/fswork/projects/rech/wqn/ufb58bn/ProjetsRemi/VLG-CBM/datasets/CUB', help='Root path to CUB dataset')
    parser.add_argument('--pth_metadata', type=str, default='metadata/imagnet_classes.json', help='Path to ImageNet classes metadata')
    parser.add_argument('--pth_metadata_parts', type=str, default='metadata/file_parts_mapping.json', help='Path to ImageNet parts metadata')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.disantangled_version == 'None':
        args.disantangled_version = False

    # Call the function to train and test the classifier
    train_and_test_clip_linear(dataset=args.dataset, cbm=args.cbm, device=device, disantangled_version=args.disantangled_version,
                               restrict_samples=args.restrict_samples, part=args.part, lr=args.lr,
                               root_monumai=args.root_monumai, root_imagenet=args.root_imagenet, root_cub=args.root_cub,
                               pth_metadata=args.pth_metadata, pth_metadata_parts=args.pth_metadata_parts)