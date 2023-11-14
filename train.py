import argparse
import numpy as np
import torch
import torch.optim as optim

import networks
import data
import utils
import timeit

data_len = 262144
parser = argparse.ArgumentParser()
parser.add_argument('-data_path', default='/mnt/data/patch_cam/pcamv1/', help="Path to PCam HDF5 files.")
parser.add_argument('-save_path', default='models/', help="Path to save trained models")
parser.add_argument('-lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('-batch_size', default=512, type=int, help="Batch size")
parser.add_argument('-n_epoch', default=20, type=int, help="Number of train iterations")
parser.add_argument('-device', default='1', type=int, help="CUDA device")
#parser.add_argument('-device_name', default='A4500', type=str, help="CUDA device")
# parser.add_argument('-save_freq', default=100000, type=int, help="Frequency to save trained models")
parser.add_argument('-visdom_freq', default=25000000, type=int, help="Frequency  plot training results")
parser.add_argument('-n', default=10, type=int, help="Number of times to run the train")
args = parser.parse_args()
print(args)


# Dataset
dataset_train = data.PatchCamelyon(args.data_path, mode='train', batch_size=args.batch_size, augment=False)
dataset_valid = data.PatchCamelyon(args.data_path, mode='valid', batch_size=args.batch_size)

# Device
device = 'cuda:{}'.format(args.device)
print(device)

# Model
model = networks.CamelyonClassifier()
model.to(device)

#torch.cuda.device(args.device)
device_name = torch.cuda.get_device_name(device=device).split(' ')[-1]
print(device_name)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8)

# Loss function
criterion = utils.loss

# Visdom writer
#writer = utils.Writer()


def train():
    def _train():
        model.train()

        n_iters = int(262144 / args.batch_size * args.n_epoch)
        #print(n_iters)


        losses = []
        for idx in range(1, n_iters + 1):

            # Zero gradient
            optimizer.zero_grad()

            # Load data to GPU
            sample = dataset_train[idx]
            images = sample['images'].to(device)
            labels = sample['labels'].to(device)

            # Forward pass
            predicted = model(images)

            # Loss
            loss = criterion(predicted, labels)
            losses.append(loss.data.item())

            # Back-propagation
            loss.backward()
            optimizer.step()

            print("Iteration: {:04d} of {:04d}\t\t Train Loss: {:.4f}".format(idx, n_iters, np.mean(losses)),
                  end="\r")

            if idx % args.visdom_freq == 0:

                # Get loss and metrics from validation set
                val_loss, accuracy, f1, specificity, precision = validation()

                # # Plot train and validation loss
                # writer.plot('loss', 'train', 'Loss', idx, np.mean(losses))
                # writer.plot('loss', 'validation', 'Loss', idx, val_loss)
                #
                # # Plot metrics
                # writer.plot('accuracy', 'test', 'Accuracy', idx, accuracy)
                # writer.plot('specificity', 'test', 'Specificity', idx, specificity)
                # writer.plot('f1', 'test', 'F1', idx, f1)
                # writer.plot('precision', 'test', 'Precision', idx, precision)

                # Print output
                print("\nIteration: {:04d} of {:04d}\t\t Valid Loss: {:.4f}".format(idx, n_iters, val_loss),
                      end="\n\n")

                # Set model to training mode again
                model.train()

        # if idx % args.save_freq == 0:
        #     torch.save(model.state_dict(), 'models/model-{:05d}.pth'.format(idx))
    return _train


def validation():
    model.eval()

    losses = []
    accuracy = []
    f1 = []
    specificity = []
    precision = []
    for idx in range(len(dataset_valid)):

        sample = dataset_valid[idx]

        # Load data to GPU
        images = sample['images'].to(device)
        labels = sample['labels'].to(device)

        # Forward pass
        predicted = model(images)

        # Loss
        loss = criterion(predicted, labels)
        losses.append(loss.data.item())

        # Metrics
        metrics = utils.metrics(predicted, labels)
        accuracy.append(metrics['accuracy'])
        f1.append(metrics['f1'])
        specificity.append(metrics['specificity'])
        precision.append(metrics['precision'])

    return torch.tensor(losses).mean(), torch.tensor(accuracy).mean(), torch.tensor(f1).mean(), \
           torch.tensor(specificity).mean(), torch.tensor(precision).mean()


if __name__ == '__main__':

    f = train()
    print(f"Number of runs: {args.n}")
    time_ls = []

    for i in range(args.n):
        print(f"running: {i}")
        time_taken = timeit.timeit(f,number=1)
        print(f"the time it takes: {time_taken}")
        time_ls.append(time_taken)

    time_array = np.array(time_ls)
    pt = 'result/' + device_name + '_run=' + str(args.n) + '.npy'
    np.save(pt,time_array)
        # avg_runtime = time_taken / args.n
        # print(f"Avg. runtime in s: {avg_runtime}")
        # avg_epoch_time = (time_taken / args.n) / args.n_epoch
        # print(f"Avg. epoch time in s: {avg_epoch_time}")
        # samples_per_sec = (len(dataset) * args.n * num_epochs) / time_taken
        # print(f"Avg. samples/s: {samples_per_sec / num_epochs}")

