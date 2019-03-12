import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


def show_and_save(img, file_name):
    r"""Show and save the image.

    Args:
        img (Tensor): The image.
        file_name (Str): The destination.

    """
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(npimg, cmap='gray')
    plt.imsave(f, npimg)


def train(model, train_loader, n_epochs=20, lr=0.01):
    r"""Train a RBM model.

    Args:
        model: The model.
        train_loader (DataLoader): The data loader.
        n_epochs (int, optional): The number of epochs. Defaults to 20.
        lr (Float, optional): The learning rate. Defaults to 0.01.

    Returns:
        The trained model.

    """
    # optimizer
    train_op = optim.Adam(model.parameters(), lr)

    # train the RBM model
    model.train()

    for epoch in range(n_epochs):
        loss_ = []
        for _, (data, target) in enumerate(train_loader):
            v, v_gibbs = model(data.view(-1, 784))
            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()

        print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))

    return model
