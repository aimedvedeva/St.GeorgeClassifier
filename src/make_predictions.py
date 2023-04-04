import torch

def make_predictions(model, data, device):
    pred_labels = []
    model.eval()
    with torch.inference_mode():
        for image in data:
            image = image.unsqueeze(dim=0).to(device)

            pred_logit = model(image)

            pred_label = pred_logit.argmax(dim=1)

            pred_labels.append(pred_label)
    return pred_labels

