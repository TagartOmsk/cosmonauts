import numpy as np
import pandas as pd
import torch
from torch import nn, optim


def train(model, dataset, batch_size, learning_rate=1e-3, epochs=5, use_cuda=False):

    device = torch.device('cuda:0' if torch.cuda.is_available() & use_cuda else 'cpu')

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    criterion = nn.L1Loss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (features, target) in enumerate(train_loader):
            features = features.to(device)
            target = target.to(device)

            out = model(features)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), './model.ckpt')

    return model
