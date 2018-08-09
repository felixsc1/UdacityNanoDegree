import argparse
import model_helpfunctions
import utility_helpfunctions
import json

# Parse input arguments


parser = argparse.ArgumentParser(description = 'Make Predictions')

parser.add_argument('input', type=str,
                    help='Image input')

parser.add_argument('checkpoint', type=str,
                    help='Model Checkpoint to use')

parser.add_argument('--top_k', type=int, default=5,
                    help='Return top k classes')

parser.add_argument('--category_names', type=str, 
                    help='json file to map categories to names')

parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='Use GPU')
parser.set_defaults(gpu=False)

args = parser.parse_args()

 
# Use GPU if selected
    
device = ('cpu')
if args.gpu:
    device = ('cuda')

    
# Load checkpoint and rebuild the model
    
model, criterion, optimizer, epochs = model_helpfunctions.load_checkpoint(
                                                            args.checkpoint)



# Make Prediction

probabilities, classes = utility_helpfunctions.predict(
                                                args.input,
                                                model,
                                                args.top_k,
                                                device)


# Print results

print(f'Probabilities: {probabilities}')
print(f'Classes: {classes}')


# Use category names if provided

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    labels = (cat_to_name[i] for i in classes)
    print(f'Flower names: {list(labels)}')
