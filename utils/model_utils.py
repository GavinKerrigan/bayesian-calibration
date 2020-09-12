import pathlib
import time
import torch

""" This file is used for various model utilities, e.g. loading a pre-trained model.
"""


def load_trained_model(model_name, train_set, device=torch.device('cpu')):
    """ Loads a pre-trained model from a state dict.

    Assumes that your models are saved in 'bayesian-calibration/models'
        and that your state dicts are saved in 'bayesian-calibration/models/checkpoints'

    Args:
        model_name: str ;
        train_set: str ;
        device: str; cpu by default
    Returns:
        A trained PyTorch model in eval mode.
    """
    print('\nLoading pre-trained model')
    print('----| Model: {}  Train set: {}'.format(model_name, train_set))

    train_set = train_set.lower()

    if train_set.startswith('cifar'):
        # Load local cifar-trained models
        num_classes = {'cifar100': 100,
                       'cifar10': 10}

        train_set = train_set.lower().strip()
        model_name = model_name.lower().strip()

        # Load the saved state dict
        path_str = 'models/checkpoints/{}_{}.tar'.format(model_name, train_set)
        checkpoint_path = pathlib.Path(path_str).resolve()
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']

        if model_name == 'resnet-110':
            from models.resnet import resnet
            state_dict = _strip_parallel_model(state_dict)
            model = resnet(num_classes=num_classes[train_set], depth=110, block_name='BasicBlock')
        elif model_name == 'alexnet':
            from models.alexnet import alexnet
            state_dict = _strip_parallel_model(state_dict)
            model = alexnet(num_classes=num_classes[train_set])
        elif model_name == 'vgg19-bn':
            from models.vgg import vgg19_bn
            state_dict = _strip_parallel_model(state_dict)
            model = vgg19_bn(num_classes=num_classes[train_set])
        else:
            raise NotImplementedError

        model.load_state_dict(state_dict)
    elif train_set == 'imagenet':
        # Thin wrapper to load PyTorch pretrained imagenet models
        import torchvision.models as models
        model = getattr(models, model_name)(pretrained=True)
    else:
        raise NotImplementedError

    model.eval()
    return model.to(device)


def forward_pass(model, data_loader, n_classes, device=torch.device('cpu')):
    """" Performs a forward pass of the model on the given data loader.

    Returns:
        logits: tensor ; shape (data_loader_size, n_classes)
        labels: tensor ; shape (data_loader_size, )
    """
    t0 = time.time()
    print('\nRunning forward pass')

    with torch.no_grad():
        data_loader_size = len(data_loader.dataset)
        batch_size = data_loader.batch_size

        # Get model logits / ground-truth labels on all of data_loader
        logits = torch.empty((data_loader_size, n_classes)).to(device)
        labels = torch.empty((data_loader_size,)).to(device)

        for i, (batch, batch_labels) in enumerate(data_loader):
            batch = batch.to(device)  # Push batch to device
            batch_logits = model(batch)  # Forward pass

            logits[i * batch_size:(i + 1) * batch_size] = batch_logits  # Store logits
            labels[i * batch_size:(i + 1) * batch_size] = batch_labels  # Store labels
        t1 = time.time()
        print('----| Forward pass complete (Runtime (s): {:.2f})'.format(t1 - t0))

    return logits, labels


def _strip_parallel_model(state_dict):
    """ Strips the state dict of a model trained with torch.nn.DataParallel, for non-parallel use.

    See: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v

    return new_state_dict
