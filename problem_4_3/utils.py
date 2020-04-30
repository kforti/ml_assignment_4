import os
import time

import torch
from torchvision.utils import save_image
from torch import nn
from torch.autograd import Variable


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x


def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


def train_autoencoder(model, criterion, optimizer, num_epochs, dataloader, name, model_path, writer):
    if not os.path.exists('./{}_img'.format(name)):
        os.mkdir('./{}_img'.format(name))

    running_loss = 0.0
    running_mse_loss = 0.0
    for epoch in range(num_epochs):

        for i, (img, _) in enumerate(dataloader):
            img = img.view(img.size(0), -1)
            img = Variable(img)
            # ===================forward=====================
            output = model(img)

            loss = criterion(output, img)
            MSE_loss = nn.MSELoss()(output, img)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        writer.add_scalar('training loss', running_loss / len(dataloader), epoch)
        writer.add_scalar('training MSE loss', running_mse_loss / len(dataloader), epoch)
        running_loss = 0.0
        running_mse_loss = 0.0
        # ===================log========================

        print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data.item(), MSE_loss.data.item()))

        if epoch % 10 == 0:
            x = to_img(img.cpu().data)
            x_hat = to_img(output.cpu().data)
            save_image(x, './{}_img/x_{}.png'.format(name, epoch))
            save_image(x_hat, './{}_img/x_hat_{}.png'.format(name, epoch))

    torch.save(model.state_dict(), model_path)
    return model


def test_autoencoder(loader, model, save_path):
    for img, labels in loader:
        img = img.view(img.size(0), -1)
        img = Variable(img)
        # ===================forward=====================
        output = model(img)
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        save_image(x, '{}/x_original.png'.format(save_path))
        save_image(x_hat, '{}/x_decoded.png'.format(save_path))
        break


def train_classifier(model, criterion, optimizer, num_epochs, dataloader, name, model_path, writer, count=None):
    if not os.path.exists('./{}_img'.format(name)):
        os.mkdir('./{}_img'.format(name))
    start = time.time()
    running_loss = 0.0
    running_correct = 0
    total_predicted = 0
    denom = len(dataloader)
    for epoch in range(num_epochs):
        c = 0
        for i, data in enumerate(dataloader):
            img, targets = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, targets)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_predicted += len(predicted)
            running_correct += (predicted == targets).sum().item()
            c += 1
            if count and c == count:
                denom = count
                break

        writer.add_scalar('training loss', running_loss / denom, epoch + 1)
        writer.add_scalar('accuracy', running_correct / total_predicted, epoch + 1)

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, running_loss / denom))
        running_loss = 0.0
        running_correct = 0
        total_predicted = 0
    end = time.time()
    print("Training Time {}: {}".format(name, end - start))
    torch.save(model.state_dict(), model_path)
    return model

def test_classifier(loader, model):
    correct_count, all_count = 0, 0
    for images, labels in loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if (true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))
