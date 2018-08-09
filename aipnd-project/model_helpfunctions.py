# Imports here
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def get_dataloader(data_dir, batch_size):

    data_transforms = {}
    data_transforms['train'] = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    data_transforms['valid'] = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    data_transforms['test'] = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    image_datasets = {}
    dataloader = {}
    batch_size = 64

    for imgtype in ['train', 'valid', 'test']:
        image_datasets[imgtype] = datasets.ImageFolder(data_dir+'/'+imgtype, transform=data_transforms[imgtype])
        dataloader[imgtype] = torch.utils.data.DataLoader(image_datasets[imgtype], batch_size=batch_size, shuffle=True)

    dataloader['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)
    
    class_to_idx = image_datasets['test'].class_to_idx
    
    return dataloader, class_to_idx

# restricted to one hidden layer
def Network(num_in_features, num_out_features, hidden_units, dropoutrate=0.5):
    Network = nn.Sequential()
    Network.add_module('fc0', nn.Linear(num_in_features, hidden_units))
    Network.add_module('relu0', nn.ReLU())
    Network.add_module('drop0', nn.Dropout(dropoutrate))
    Network.add_module('outputlayer', nn.Linear(hidden_units, num_out_features))
    Network.add_module('softmax', nn.LogSoftmax(dim=1))
    return Network




def get_model(arch, hidden_units, learning_rate, class_to_idx):
    
    if arch == 'vgg':
        model = models.vgg11_bn(pretrained=True)
    elif arch == 'densenet':
        model = models.densenet161(pretrained=True)
    else:
        raise RuntimeError("model not supported")
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = Network(25088, 102, hidden_units)
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer


#%% Model training
    

def validation(model, dataloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in dataloader:

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss/len(dataloader), accuracy/len(dataloader)




def train_model(model, epochs, device, dataloader, optimizer, criterion):

    steps = 0
    running_loss = 0
    print_every = 40

    model.to(device)

    running_loss = 0
    for e in range(epochs):
        model.train()
        
        for images, labels in dataloader['train']:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
                
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloader['valid'], criterion, device)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss),
                      "Validation Accuracy: {:.3f}".format(accuracy))
                
                running_loss = 0       
                model.train()


#%% Saving a checkpoint
    
def save_checkpoint(file_path, model, optimizer, 
                    arch, learning_rate, hidden_units, epochs):
    
    checkpoint = {'architecture': arch,
                  'hidden_units': hidden_units,
                  'learning_rate': learning_rate,
                  'classifier_state_dict': model.classifier.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epochs,
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, file_path)



#%% Loading a checkpoint

def load_checkpoint(file_path):

    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
    
    # Rebuilding the model, criterion, and optimizer
    model, criterion, optimizer = get_model(checkpoint['architecture'],
                                            checkpoint['hidden_units'],
                                            checkpoint['learning_rate'],
                                            checkpoint['class_to_idx'])
    
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    epochs = checkpoint['epoch']
    
    return model, criterion, optimizer, epochs
