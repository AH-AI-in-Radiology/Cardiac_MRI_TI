import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from IPython.display import clear_output
from torch.utils.data import Dataset, DataLoader
import json

def weighted_bceloss_logits(pred, y, decay=1):
    weight = 1/(decay*torch.abs(y))
    weight[weight>3] = 3

    y[y>0] = 1
    y[y==0] = 0.5
    y[y<0] = 0

    return (nn.BCEWithLogitsLoss(reduction='none')(pred, y) * weight).mean()

def save_parameters(model, optimiser, epoch, directory, name=None, tag_epoch_number=True):
    try: 
        os.listdir(directory)
    except:
        os.makedirs(directory)
    
    saved_models = os.listdir(directory)
    saved_models.sort()
    
    checkpoint = {
        'model': model.state_dict(),
        'optimiser': optimiser.state_dict()
    }
    if not isinstance(name, str):
        name = 'model'
    if tag_epoch_number:
        suffix = f"_{str(epoch).zfill(6)}.pth"
    else:
        suffix = ".pth"
    
    torch.save(checkpoint, f"{directory}/{name}{suffix}")

def load_parameters(model, optimiser=None, directory=None, name=None, feature_extractor_only=False, load_index=-1):
    if directory == None:
        directory = f"./{model.name}"
    
    try: 
        os.listdir(directory)
    except:
        os.makedirs(directory)
    
    saved_models = os.listdir(directory)
    
    if len(saved_models) != 0:
        if name:
            model_dir = f"{directory}/{name}"
        else:
            _, models = zip(*sorted([(x[-10:-4], x) for x in saved_models if '.pth' in x]))
            model_dir = f"{directory}/{models[load_index]}"
        
        checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
        
        if feature_extractor_only:
            keys = list(checkpoint['model'].keys())
            for k in keys:
                if 'feature_extractor' != k[:len('feature_extractor')]:
                    del checkpoint['model'][k]
        model.load_state_dict(checkpoint['model'], strict=False)
        
        if optimiser != None:
            try: 
                optimiser.load_state_dict(checkpoint['optimiser'])
            except:
                print('Optimiser unable to be loaded')
        
        print(model_dir+' loaded')
    else:
        print(f'Starting new in {directory}')

def save_config(config, path):
    try: 
        os.listdir(path)
    except:
        os.makedirs(path)
    with open(f"{path}/config.json", 'w') as f:
        json.dump(config, f)
