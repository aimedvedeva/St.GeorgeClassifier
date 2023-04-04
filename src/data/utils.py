import os
from math import floor

import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.data.GeorgeDataset import GeorgeDataset
from sklearn.model_selection import train_test_split


def label_and_merge(project_path, raw_data_path, processed_data_path, test_size):
    # read classes' data
    george_data = pd.read_csv(os.sep.join([project_path, raw_data_path, "george", "georges.csv"]), names=["path"])
    no_george_data = pd.read_csv(os.sep.join([project_path, raw_data_path, "no_george", "non_georges.csv"]), names=["path"])

    # save only image names without links for both classes
    george_data['label'] = 1
    george_data['path'] = george_data['path'].apply(
        lambda x: os.sep.join([project_path, raw_data_path, "george", x.split('/')[-1]]))

    no_george_data['label'] = 0
    no_george_data['path'] = no_george_data['path'].apply(
        lambda x: os.sep.join([project_path, raw_data_path, "no_george", x.split('/')[-1]]))

    # cut 0.2 from the first class
    amount = floor(test_size * len(george_data))
    test = george_data.iloc[0:amount,:]
    george_data = george_data.iloc[amount:,:]

    # cut 0.2 from the second class
    amount = floor(test_size * len(no_george_data))
    test = pd.concat([test, no_george_data.iloc[0:amount,:]], axis=0, ignore_index=True)
    no_george_data = no_george_data.iloc[amount:,:]

    # save to test sample
    test.to_csv(os.sep.join([project_path, processed_data_path, "test.csv"]), index=False)

    # save data for future train and validation samples
    data = pd.concat([george_data, no_george_data], axis=0, ignore_index=True)
    data.to_csv(os.sep.join([project_path, raw_data_path, 'data.csv']), index=False)

def generate_samples(project_path, raw_data_path, processed_data_path, validate_size):

    data = pd.read_csv(os.sep.join([project_path, raw_data_path, "data.csv"]))

    X_train, X_val, y_train, y_val = train_test_split(data['path'],
                                                      data['label'],
                                                      test_size=validate_size,
                                                      stratify=data['label'])

    # save train sample
    train = pd.DataFrame({'path': X_train, 'label':y_train})
    train.to_csv(os.sep.join([project_path, processed_data_path, "train.csv"]), index=False)

    # save validate sample
    validate = pd.DataFrame({'path': X_val, 'label':y_val})
    validate.to_csv(os.sep.join([project_path, processed_data_path, "validate.csv"]), index=False)

def create_dataloaders(project_path, data_path, batch_size, manual_transforms):

    train_dataset = GeorgeDataset(os.sep.join([project_path, data_path, "train.csv"]), manual_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate_dataset = GeorgeDataset(os.sep.join([project_path, data_path, "validate.csv"]), manual_transforms)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = GeorgeDataset(os.sep.join([project_path, data_path, "test.csv"]), manual_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validate_dataloader, test_dataloader


def get_sample_to_predict(project_path, predict_data_path, manual_tranforms):
    images = []
    paths = []
    for file in os.listdir(os.path.join(project_path, predict_data_path)):
        image_path = os.fsdecode(os.path.join(project_path, predict_data_path, file))
        paths.append(image_path)
        img = Image.open(image_path).convert('RGB')
        img_tensor = manual_tranforms(img)
        images.append(img_tensor)
    return images, paths

def plot_predictions(images_paths, pred_labels, project_path, predicted_data_path):
    for i, image_path in enumerate(images_paths):
        plt.figure()

        plt.imshow(Image.open(image_path).convert('RGB'))

        prediction_title = 'george' if pred_labels[i].item() == 1 else 'non_george'
        title_text = f"Pred: {prediction_title}"

        plt.title(title_text, fontsize=10, c="g")

        plt.savefig(os.path.join(project_path, predicted_data_path, 'predicted_'+image_path.split('\\')[-1]))