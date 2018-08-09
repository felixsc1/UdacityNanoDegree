import argparse
import model_helpfunctions
import os


# Parse input arguments
    
parser = argparse.ArgumentParser(description = 'Train a model')

parser.add_argument('data', type=str,
                    help='data directory')
parser.add_argument('--save_dir', type=str, default='', 
                    help='save directory')
parser.add_argument('--arch', type=str, default='vgg',
                    help='architecture')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512,
                    help='Hidden units')
parser.add_argument('--epochs', type=int, default=8,
                    help='Epochs')
parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='Use GPU')
parser.set_defaults(gpu=False)

args = parser.parse_args()


# Get the data

dataloaders, class_to_idx = model_helpfunctions.get_dataloader(args.data,
                                                               args.epochs)


# Create the model

model, criterion, optimizer = model_helpfunctions.get_model(args.arch,
                                                            args.hidden_units,
                                                            args.learning_rate,
                                                            class_to_idx)

# Use GPU if selected

device = ('cpu')
if args.gpu:
    device = ('cuda')


# Train the model

model_helpfunctions.train_model(model,
                                args.epochs,
                                device,
                                dataloaders,
                                optimizer,
                                criterion)

# Save checkpoint

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

file_path = os.path.join(args.save_dir,'checkpoint.pth')

model_helpfunctions.save_checkpoint(file_path,
                                        model,
                                        optimizer,
                                        args.arch,
                                        args.learning_rate,
                                        args.hidden_units,
                                        args.epochs)

