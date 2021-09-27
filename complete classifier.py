import torch
from torch import optim, nn
# import visdom
import torchvision
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,wide_resnet101_2,mobilenet_v2




def evalute(model, loader, device):
    model.eval()
    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(epoch, device, train_loader, val_loader, total,train_state=None,resume=False):
    model = resnet101().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    train_loss_list = []
    acc_val_list = []
    starting_epoch = 0
    total_epoch = epoch
    if resume:
        checkpoint_path = "./model/model_" + epoch + ".pt"
        train_state = torch.load(checkpoint_path)
    if train_state is not None:
        model.load_state_dict(train_state['model_state_dict'])
        optimizer.load_state_dict(train_state['optimizer_state_dict'])
        starting_epoch = train_state['epoch'] + 1

    for epoch in range(starting_epoch, epoch):
        with tqdm(total=total, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for step, (x, y) in enumerate(train_loader):
                # print(x.size(),y.size())
                x, y = x.to(device), y.to(device)
                # print(x.size(),y.size())

                model.train()
                logits = model(x)
                loss = criteon(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # viz.line([loss.item()], [global_step], win='loss', update='append')
                global_step += 1
                train_loss_list.append(loss.item())
                # print('epoch: ', epoch, 'step:', step, 'loss: ', loss.item())

                pbar.set_postfix(**{'loss': loss.item(),"lr" : get_lr(optimizer)})
                pbar.update(1)

            val_acc = evalute(model, val_loader, device)
            acc_val_list.append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
            print(f"\nthe epoch{epoch + 1} accuracy is {val_acc}")
            print(f"best_accuracy(epoch{best_epoch + 1}) is {best_acc}")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            # torch.save(model.state_dict(), 'best.mdl')
            model_path = "./model/" + "epoch{:03d} loss_{} acc_{}.pt".format(epoch + 1,loss.item(),val_acc)
            torch.save(checkpoint, model_path)
        # if epoch % 1 == 0:
        #     val_acc = evalute(model, val_loader, device)
        #     acc_val_list.append(val_acc)
        #     if val_acc > best_acc:
        #         best_epoch = epoch
        #         best_acc = val_acc
        #         checkpoint = {
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #         }
        #         # torch.save(model.state_dict(), 'best.mdl')
        #         model_path = "./model/" + "epoch_{:03d} loss_.pt".format(epoch,)
        #         torch.save(checkpoint, model_path)

    # print('best acc:', best_acc, 'best epoch:', best_epoch)

    # model.load_state_dict(torch.load('best.mdl'))
    # print('loaded from ckpt!')


if __name__ == '__main__':

    epoch = 100
    batchsz = 100
    lr = 1e-3
    device = torch.device('cuda:0')
    torch.manual_seed(1234)

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transforms.ToTensor())

    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transforms.ToTensor())



    train_loader = DataLoader(train_set, batch_size=batchsz, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batchsz, num_workers=8)


    train(epoch, device, train_loader, val_loader,total = len(train_loader))

