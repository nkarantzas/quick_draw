import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from load_tensor_data import class_encoding
from flags import FLAGS
from scoring_functions import get_batch_top3score


def train_function(model, optimizer, loss_fn, device, sample):
    data = sample['image'].to(device)
    target = sample['label'].to(device)
    optimizer.zero_grad()
    # Forward pass
    output = model(data)
    loss = loss_fn(output, target)
    # Backprop and optimize
    loss.backward()
    optimizer.step()
    return target, output, loss


def validation_function(model, val_loader, loss_fn, device):
    model.eval()
    score = 0
    total = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            data = sample['image'].to(device)
            target = sample['label'].to(device)
            output = model(data)
            val_loss = loss_fn(output, target).item()
            # calculate mean average precision @3
            score += get_batch_top3score(target, output)
            total += target.size(0)
    return score * 100 / total, val_loss


def validation_summary(model, device, val_loader, num_classes):
    # Test the model
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, sample in enumerate(val_loader):
            data = sample['image'].to(device)
            target = sample['label'].to(device)
            output = model(data)
            _, predictions = torch.max(output, dim=1)
            total += target.size(0)
            correct += (predictions == target).sum().item()
            for t, p in zip(target.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        val_overall_accuracy = 100 * correct / total
        val_average_accuracy = np.mean((confusion_matrix.diag() / confusion_matrix.sum(1)).numpy() * 100)
        confusion_matrix_np = confusion_matrix.numpy()
        print('val oa: %f, val aa: %f' % (val_overall_accuracy, val_average_accuracy))
        print('confusion matrix:')
        print(confusion_matrix_np)

    return val_overall_accuracy, val_average_accuracy, confusion_matrix_np


def test_function(model, test_loader, device, length_test, batch_size):
    # Test the model
    model.eval()
    predictions = np.zeros((length_test, 4), dtype=object)
    # get the class encoding for submission purposes
    current_enc = class_encoding(FLAGS.training_data)
    enc_stripped = [key.replace(" ", "_") for key in list(current_enc.keys())]
    enc2word = np.vectorize(lambda x: enc_stripped[x])

    with torch.no_grad():
        for i, sample in tqdm(enumerate(test_loader)):
            data = sample['image'].to(device)
            key_ids = sample['key_id'].reshape(data.size(0), 1)
            key_ids = key_ids.cpu().data.numpy().astype(object)
            output = model(data)

            # preds will be a tensor array of integer class predictions
            _, preds = torch.topk(output, k=3, dim=1)
            # enc2word takes the predictions and gives a numpy array of top3 word predictions
            preds = enc2word(preds.cpu().data.numpy()).astype(object)
            predictions[i * batch_size: i * batch_size + data.size(0)] = np.concatenate([key_ids, preds], axis=1)

    # merge predictions to create submission file
    guesses = (predictions[:, 1] + ' ' + predictions[:, 2] + ' ' + predictions[:, 3]).reshape(length_test, 1)
    submission = np.concatenate([predictions[:, 0].reshape(length_test, 1), guesses], axis=1)
    return pd.DataFrame(submission, columns=['key_id', 'word'])


