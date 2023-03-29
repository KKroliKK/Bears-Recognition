import os
import time
import copy
import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from typing import List, Tuple, Dict
from torchvision import models, transforms
from IPython.display import clear_output
from torch.utils.data import DataLoader

SEED = 42
np.random.seed(SEED)
# random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(
        model: nn.Module, 
        dataloaders: Dict[str, DataLoader], 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        num_epochs: int=25, 
        is_inception: bool=True
    ) -> Tuple[nn.Module, List[float]]:
    '''Trains given PyTorch model.

        Args:
            model: PyTorch model to train.
            dataloaders: Dictionary with dataloaders in format: 
                {'train': train_loader, 'val': valid_loader}
            criterion: PyTorch criterion to calculate loss with.
            optimizer: PyTorch optimizer to change model's parameters.
            num_epochs: Num epochs to train the model.
            is_inception: If true then calculates inception specific loss while training.
        
        Returns:
            Trained model and history of validation accuracy changing.
    '''
    model.to(device)
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} of {num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def initialize_inception_model(
        num_classes: int, 
        num_unfreezed: int=None, 
        use_pretrained: bool=True
    ) -> Tuple[nn.Module, int]:
    '''Returns pretrained on ImageNet inception model with chosen number of unfreezed layers.

    Args:
        num_classes: Number of heads of the classifier.все 
        num_infreezed: Number of lower levels to set .requires_grad = True.
        use_pretrained: If true then loads pretrained on ImageNet weights else 
            initializes random ones.

    Returns:
        Inception model and input size expected by it.
    '''
    model = models.inception_v3(pretrained=use_pretrained)
    num_unfreezed = -num_unfreezed if num_unfreezed != None else num_unfreezed
    for param in list(model.parameters())[:num_unfreezed]:
        param.requires_grad = False

    # Handle the auxilary net.
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

    # Handle the primary net.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 299

    clear_output()
    print('Inception model was loaded sucessfully.')

    return model, input_size


def test_model(
        model: torch.nn.Module, 
        dataloader: DataLoader
    ) -> Tuple[List[int], List[int]]:
    '''Makes predictions on test data in evaluation mode of the given model.

    Args:
        model: Trained pytorch model to use for prediction.
        dataloader: pytorch dataloader with test data.

    Returns:
        Two lists with true and predicted labels.
    '''
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        with torch.no_grad():
            outp = model(x_batch)

        preds = outp.argmax(-1)

        y_true += y_batch.tolist()
        y_pred += preds.tolist()

    return y_true, y_pred


def predict_sample(
        model: torch.nn.Module,
        input_size: int,
        path: str
    ) -> int:
    '''Predicts label for a given sample with given model.

    Args:
        model: Trained pytorch model to use for prediction.
        input_size: Size to reshape incoming image according 
            to the model architecture.
        path: Path to the image to predict on.

    Returns:
        Predicted label.
    '''
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(path).convert('RGB')
    batch = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(batch.to(device)).squeeze(0)
    pred_label = np.argmax(out.tolist())

    return pred_label


def realname(path: str, root: str=None) -> str:
    '''Returns real name of the folder for given path to it.
    '''
    if root is not None:
        path = os.path.join(root, path)
    result = os.path.basename(path)
    if os.path.islink(path):
        realpath = os.readlink(path)
        result= '%s -> %s' % (os.path.basename(path), realpath)
    return result


def ptree(startpath:str, depth: int=2, show_files: bool=False) -> None:
    '''Prints directory tree with specified depth.

    Args:
        startpath: Directory from where to start printing the tree.
        depth: Max epth of the printed tree.
        show_files: If true prints content of the dirictiries else prints only
            folders' names.
    '''
    prefix=0
    if startpath != '/':
        if startpath.endswith('/'): startpath=startpath[:-1]
        prefix=len(startpath)
    for root, dirs, files in os.walk(startpath):
        level = root[prefix:].count(os.sep)
        if depth >-1 and level > depth: continue
        indent=subindent =''
        if level > 0:
            indent = '|   ' * (level-1) + '|-- '
        subindent = '|   ' * (level) + '|-- '
        print('{}{}'.format(indent, realname(root)))
        for d in dirs:
            if os.path.islink(os.path.join(root, d)):
                print('{}{}'.format(subindent, realname(d, root=root)))
        if show_files:                
            for f in files:
                print('{}{}'.format(subindent, realname(f, root=root)))