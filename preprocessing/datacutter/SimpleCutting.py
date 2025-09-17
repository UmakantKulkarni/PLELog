from CONSTANTS import *


def cut_by_613(instances):
    # Ensure the semi-supervised setting keeps labeled normal sequences in the
    # training split while unlabeled data (mixture of normal and anomalous
    # samples) follow the usual 60/10/30 partition for train/dev/test.
    labeled = [inst for inst in instances if getattr(inst, 'is_labeled', False)]
    unlabeled = [inst for inst in instances if not getattr(inst, 'is_labeled', False)]

    if unlabeled:
        np.random.shuffle(unlabeled)
        train_unlabeled = int(0.6 * len(unlabeled))
        dev_unlabeled = int(0.1 * len(unlabeled))

        train = labeled + unlabeled[:train_unlabeled]
        dev = unlabeled[train_unlabeled:train_unlabeled + dev_unlabeled]
        test = unlabeled[train_unlabeled + dev_unlabeled:]
    else:
        train = list(labeled)
        dev = []
        test = []

    if train:
        np.random.shuffle(train)
    if dev:
        np.random.shuffle(dev)
    if test:
        np.random.shuffle(test)

    return train, dev, test
