import torch
import numpy as np
from tqdm import tqdm
import random

def partition_ids(ids: list[int], main_bin_num=10):
    n = len(ids)
    bin_size = n // main_bin_num
    indexes = np.arange(n)
    np.random.shuffle(indexes)
    binned = []
    for i in range(main_bin_num):
        one_bin = ids[indexes[i:i+bin_size]]
        binned.append(one_bin)

    return binned

def partition_ids2(ids: list[int], img_vecs, main_bin_num=10):
    big_img_vecs = []
    big_ids = []
    leftover_ids = []
    for i in range(len(ids)):
        if np.array(img_vecs[i]).size >= 2000:
            big_img_vecs.append(img_vecs[i])
            big_ids.append(ids[i])
        else:
            leftover_ids.append(ids[i])
    
    big_ids = np.array(big_ids)
    big_img_vecs = np.array(big_img_vecs)
    big_img_vecs = big_img_vecs.reshape(big_img_vecs.shape[0], -1)

    bins = []
    for i in range(main_bin_num):
        _, indexes, num = find_independent_set(big_img_vecs, k=200)
        print(f'num: {num}')

        bins.append(big_ids[indexes])
        mask = np.ones(len(big_ids), dtype=bool)
        mask[indexes] = False
        big_ids = big_ids[mask]
        big_img_vecs = big_img_vecs[mask]

    leftover_ids.extend(big_ids)
    random.shuffle(leftover_ids)
    iter = 0
    for i in range(main_bin_num):
        while len(bins[i]) < 2000:
            bins[i] = np.append(bins[i], leftover_ids[iter])
            iter += 1

    return bins

if __name__ == '__main__':
    ids  = np.arange(100, 200)
    partitioned = partition_ids(ids)
    print(partitioned[3])

def is_linearly_independent(set_of_vectors, vector):
    # Check if the new vector is linearly independent of the set
    coefficients = np.linalg.lstsq(set_of_vectors, vector, rcond=None)[0]
    return np.linalg.norm(vector - np.dot(set_of_vectors, coefficients)) > 1e-10

def find_independent_set(vectors, k=200):
    vectors = np.array(vectors)
    n = len(vectors)
    independent_set = vectors[:k]
    indexes = np.arange(k)
    i = 1
    for j, vector in enumerate(tqdm(vectors)):
        if i >= k:
            return np.array(independent_set), indexes, i
        if n - j < k - i:
            return np.array(independent_set), indexes, i
        if is_linearly_independent(np.array(independent_set[:i]).T, vector):
            independent_set[i] = vector
            indexes[i] = j
            i += 1
    return np.array(independent_set), indexes, i

def validate(criterion, loader, net):
    net.eval()

    with torch.no_grad():
        true = []
        pred = []
        for x, y in loader:
            y_hat = net(x)
            true.append(y.numpy())
            pred.append(y_hat.numpy())

        true = torch.tensor(np.concatenate(true, axis=0))
        pred = torch.tensor(np.concatenate(pred, axis=0))

        loss = criterion(pred, true)

    return loss

def l1_reg(model, reg_lambda):
    l1_regularization = torch.tensor(0., device=model.parameters().__next__().device)
    for param in model.parameters():
        l1_regularization += torch.norm(param, p=1)
    return reg_lambda * l1_regularization

def train(epochs, optim, criterion, regularise, trainloader, valloader, net, empty_net, reg_lambda=None):
    best_val_loss = np.inf
    best_net = empty_net

    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = 0

        progress_bar = tqdm(trainloader)

        for iter, (x, y) in enumerate(progress_bar):
            net.train()
            optim.zero_grad()

            y_hat = net(x)
            loss = criterion(y_hat, y)
            if reg_lambda is not None:
                loss += regularise(net, reg_lambda)
            loss.backward()
            optim.step()

            batch_loss = loss.item()
            train_loss += batch_loss
            if iter % 20 == 0:
                progress_bar.set_description(f"train | loss: {batch_loss:.4f}")

        train_loss /= len(trainloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}")

        net.eval()

        val_loss = validate(criterion, valloader, net)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_net.load_state_dict(net.state_dict())

        print(f"Epoch [{epoch + 1}/{epochs}], Val Loss: {val_loss:.4f}")

    return net, best_net

def lin_augment_affine(xs, ys, new_for_each_pair=3):
    new_x = []
    new_y = []
    for i, (x1, y1) in tqdm(enumerate(zip(xs, ys))):
        for j, (x2, y2) in enumerate(zip(xs, ys)):
            if i != j:
                weights = np.random.uniform(-1, 2, size=new_for_each_pair)
                for w in weights:
                    x_new = w*x1 + (1-w)*x2
                    y_new = w*y1 + (1-w)*y2
                    new_x.append(x_new)
                    new_y.append(y_new)
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    return np.concatenate((xs, new_x), axis=0), np.concatenate((ys, new_y), axis=0)
