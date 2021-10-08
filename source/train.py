from __future__ import print_function # future proof
import argparse
import sys
import os
import json

import pandas as pd
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# import model
from model import SimpleNet


def model_fn(model_dir):
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(model_info['input_dim'], 
                      model_info['hidden_1'],
                      model_info['hidden_2'], 
                      model_info['hidden_3'], 
                      model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)


# Load the training data from a csv file
def _get_train_loader(batch_size, data_dir):
    print("Get train loader.")

    # read in csv file
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, names=None)

    # labels are first column
    train_y = torch.from_numpy(train_data[[0]].values).float()
    # features are the rest
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    # create dataset
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

# Load the training data from a csv file
def _get_validation_loader(batch_size, data_dir):
    print("Get validation loader.")

    # read in csv file
    val_data = pd.read_csv(os.path.join(data_dir, "validation.csv"), header=None, names=None)

    # labels are first column
    val_y = torch.from_numpy(val_data[[0]].values).float()
    # features are the rest
    val_x = torch.from_numpy(val_data.drop([0], axis=1).values).float()

    # create dataset
    val_ds = torch.utils.data.TensorDataset(val_x, val_y)

    return torch.utils.data.DataLoader(val_ds, batch_size=batch_size)


# Provided train function
def train(model, train_loader, validation_loader, epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training. 
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    valid_loss_min = np.Inf # track change in validation loss
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        valid_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # prep data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # zero accumulated gradients
            # get output of SimpleNet
            output = model(data)
            # calculate loss and perform backprop
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()

        ######################    
        # validate the model #
        ######################
        model.eval()
 
        for batch_idx, (data, target) in enumerate(validation_loader, 1):
            # prep data
            data, target = data.to(device), target.to(device)
            # get output of SimpleNet
            output = model(data)
            # calculate loss and perform backprop
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)


        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, total_loss/len(train_loader), valid_loss/len(validation_loader)))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  \
                   Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            # Set model back to device after saving
            model.to(device)
            valid_loss_min = valid_loss        
        
    # save after all epochs
    save_model(model, args.model_dir)


# Provided model saving functions
def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)
    
def save_model_params(model, model_dir):
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'hidden_1': args.hidden_1,
            'hidden_2': args.hidden_2,
            'hidden_3': args.hidden_3,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)


## TODO: Complete the main code
if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
  
    ## TODO: Add args for the three model parameters: input_dim, hidden_dim, output_dim
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=2, metavar='IN',
                        help='Number of input features to model (default: 2)')
    parser.add_argument('--hidden_1', type=int, default=10, metavar='H1',
                        help='hidden dimensions of model (default: 10)')
    parser.add_argument('--hidden_2', type=int, default=10, metavar='H2',
                        help='hidden dimensions of model (default: 10)')
    parser.add_argument('--hidden_3', type=int, default=10, metavar='H3',
                        help='hidden dimensions of model (default: 10)')
    parser.add_argument('--output_dim', type=int, metavar='O', 
                        help='number of outputs (default: 1)')

    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:'.format(device))
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # get train loader
    train_loader = _get_train_loader(args.batch_size, args.data_dir) # data_dir from above..
    # get valid loader
    validation_loader = _get_validation_loader(args.batch_size, args.data_dir) # data_dir from above..

    
    
    
    ## TODO:  Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = SimpleNet(args.input_dim, args.hidden_1, args.hidden_2, args.hidden_3, args.output_dim).to(device)
    
    # Given: save the parameters used to construct the model
    save_model_params(model, args.model_dir)

    ## TODO: Define an optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    
    # Trains the model (given line of code, which calls the above training function)
    # This function *also* saves the model state dictionary
    train(model, train_loader, validation_loader, args.epochs, optimizer, criterion, device)
    
