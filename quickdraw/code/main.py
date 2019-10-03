from glob import glob
import os
import h5py
import resnet_one
import torch
from tqdm import tqdm
import torch.nn as nn
from flags import FLAGS
import json
from get_dataloaders import get_data
from training_functions import train_function, validation_function, validation_summary, test_function
from scoring_functions import get_batch_top3score


def main():
    # release gpu memory
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the list of csv training files
    training_csv_files = glob(FLAGS.training_data + "*.csv")
    test_csv = glob(FLAGS.test_data + "*.csv")[0]

    # build training, validation, and test data loaders
    print(' Preparing the data!')
    train_loader, val_loader, test_loader, test_length, weights = \
        get_data(training_csv_files[0: FLAGS.num_classes], test_csv)

    # instantiate the model
    model = getattr(resnet_one, 'resnet18')(num_classes=FLAGS.num_classes,
                                            input_channels=FLAGS.input_channels)

    # choose an optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # distribute the model to all available devices
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    # define your loss function.
    loss_function = nn.CrossEntropyLoss(reduction='mean', weight=weights.to(device))

    # initialize training and validation log lists
    val_scores = []
    val_losses = []
    train_scores = []
    train_losses = []

    print(' Started training!')
    for epoch in range(1, FLAGS.num_epochs + 1):
        # training mode. This needs to be reset after every epoch
        model.train()
        for batch_idx, sample in tqdm(enumerate(train_loader)):

            target, output, loss = train_function(model, optimizer, loss_function, device, sample)
            # print training and validation scores with respect to log intervals
            if batch_idx == 1000:
                break
            if (batch_idx + 1) % FLAGS.log_interval == 0 or (batch_idx + 1) == len(train_loader):

                # get score and loss of training batch and log them
                train_score = get_batch_top3score(target, output) * 100 / target.size(0)
                train_loss = loss.item()
                train_scores.append(train_score)
                train_losses.append(train_loss)

                # get score and loss of validation set and log them too
                val_score, val_loss = validation_function(model, val_loader, loss_function, device)
                val_scores.append(val_score)
                val_losses.append(val_loss)

                # print scores
                print(' epoch: {} ({:.2f}%): '
                      'batch loss {:.5f}, '
                      'batch score {:.2f}, '
                      'val score {:.2f}'
                      .format(epoch, 100 * (batch_idx + 1) / len(train_loader),
                              train_loss, train_score, val_score))
                model.train()  # reset back to train mode

    print('Training completed. Now testing!')
    submission = test_function(model, test_loader, device, test_length, FLAGS.batch_size)
    print('Saving logs!')

    if os.path.exists(FLAGS.save_path + '/models'):
        print('The Model-saving directory already exists')
    else:
        print('creating a Model-saving directory in your designated saving folder')
        os.mkdir(FLAGS.save_path + '/models')
    i = 0
    while os.path.exists(FLAGS.save_path + '/models/model%s.pt' % i):
        i += 1
    # saving the model
    torch.save({'epoch': FLAGS.num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               FLAGS.save_path + '/models/model%s.pt' % i)
    # save the submission file
    submission.to_csv(FLAGS.save_path + '/models/submission%s.csv' % i, index=False)
    # save logs and hyperparameters

    logs = h5py.File(FLAGS.save_path + '/models/logs%s.h5' % i, 'w')
    logs.create_dataset('train_scores', data=train_scores)
    logs.create_dataset('train_losses', data=train_losses)
    logs.create_dataset('validation_scores', data=val_scores)
    logs.create_dataset('validation_losses', data=val_losses)
    logs.create_dataset('Flags', data=json.dumps(vars(FLAGS)))
    logs.close()


if __name__ == '__main__':
    main()
