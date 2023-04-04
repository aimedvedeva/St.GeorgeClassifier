import os
import torch
from torch import nn
from torchvision import transforms
from config.config import load_config
from src.logging_utils import create_writer, save_model
from src.data.utils import create_dataloaders, get_sample_to_predict, \
    plot_predictions, label_and_merge, generate_samples
from src.evaluate_model import eval_model
from src.make_predictions import make_predictions
from src.metrics import accuracy_fn
from src.model.tiny_vgg import tiny_vgg
import numpy as np
from src.train_and_validate import train

def set_seed():
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

if __name__ == '__main__':
    set_seed()
    config = load_config(os.path.dirname(__file__), "config.yaml")
    project_path = os.path.dirname(__file__)

    # create labeled dataset and extract test sample
    # enough to use once
    #label_and_merge(project_path,
    #                config['raw_data_path'],
    #                config['processed_data_path'],
    #                config['test_size'])

    # create train and validate samples
    #generate_samples(project_path,
    #                config['raw_data_path'],
    #                config['processed_data_path'],
    #                config['validate_size'])

    # set basic trasformations for train/validate/test samples
    manual_transforms = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((config['normalize_mu'],), (config['normalize_sigma'],))
    ])

    train_dataloader, validate_dataloader, test_dataloader = create_dataloaders(project_path,
                                                                                config['processed_data_path'],
                                                                                config['batch_size'],
                                                                                manual_transforms)
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if (config['mode'] == 'load_and_eval'):
        model = tiny_vgg(in_channels=config['image_channels'], num_classes=config['classes'])
        model.load_state_dict(torch.load(os.path.join(project_path, config['models_path'], config['model_to_load_path'])))
        loss_fn = nn.CrossEntropyLoss()
        model_results = eval_model(model=model, data_loader=test_dataloader,
                                     loss_fn=loss_fn, accuracy_fn=accuracy_fn)
        print(model_results)
    elif (config['mode'] == 'train_and_validate'):
        n_epochs = config['epochs']
        model = tiny_vgg(in_channels=config['image_channels'], num_classes=config['classes']).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=config['lr'])

        writer = create_writer(experiment_name=config['experiment_name'],
                               model_name=config['model_name'],
                               extra=str(config['epochs']) + '_epochs_batch' + str(config['batch_size']))

        train(model,
              train_dataloader,
              validate_dataloader,
              optimizer,
              loss_fn,
              n_epochs,
              device,
              writer,
              config['checkpoints_to_save'],
              config['checkpoints_path'])

        save_model(model, experiment_name=config['experiment_name'],
                   model_name=config['model_name'],
                   extra=str(config['epochs']) + '_epochs_batch' + str(config['batch_size']))
    elif (config['mode'] == 'make_predictions'):
        images, images_paths = get_sample_to_predict(project_path, config['predict_data_path'], manual_transforms)
        model = tiny_vgg(in_channels=config['image_channels'], num_classes=config['classes'])
        model.load_state_dict(torch.load(os.path.join(project_path, config['models_path'], config['model_to_load_path'])))
        pred_labels = make_predictions(model, images, device)
        plot_predictions(images_paths, pred_labels, project_path, config['predicted_data_path'])