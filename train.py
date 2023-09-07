# Python 3.10.9
# 本程序实现模型训练的整个过程并保存实验结果
from LoadData import combine_img_and_label, STSDataset
from Network import DualInputNet, SingleInputNet
import torch
from torch import nn
from torchvision import transforms
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

random.seed(5)
train_rawdata = combine_img_and_label(r"D:\Data\STS\preprocess\train_data", 'train_label.xlsx')
test_rawdata = combine_img_and_label(r"D:\Data\STS\preprocess\test_data", 'test_label.xlsx')
test_rawdata = [t for t in test_rawdata if t[2] != '-1']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])
train_set = STSDataset(train_rawdata, transform)  # {1: 934, 0: 558}
test_set = STSDataset(test_rawdata, transform)  # {1: 1014, 0: 378}

batch_size = 8
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

net = DualInputNet()
nn.init.xavier_uniform_(net.combination.weight)
# nn.init.xavier_uniform_(net.combination[3].weight)
# all_param_names = [name for name, param in net.named_parameters()]
# fc_names = ['combination.0.weight', 'combination.0.bias', 'combination.3.weight', 'combination.3.bias']
fc_names = ['combination.weight', 'combination.bias']
params_conv = [param for name, param in net.named_parameters() if name not in fc_names]
params_fc = [param for param in net.combination.parameters() if param.requires_grad is True]


def train_epoch(net, x1, x2, y, loss, trainer, device):
    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(x1, x2)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum().item()
    y_ = pred.argmax(dim=-1)
    train_acc_sum = (y_ == y).sum()
    num_instance = y.numel()
    return train_loss_sum, train_acc_sum, num_instance


def eval_batch(net, test_iter, loss, device):
    eval_loss_epoch = eval_acc_epoch = num_instance = 0
    for i, (x1, x2, y) in enumerate(test_iter):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        net.eval()
        pred = net(x1, x2)
        l = loss(pred, y)
        eval_loss_epoch += l.sum().item()
        y_ = pred.argmax(dim=-1)
        eval_acc_epoch += (y_ == y).sum()
        num_instance += y.numel()
    return eval_loss_epoch / num_instance, eval_acc_epoch / num_instance


def train(net, train_iter, test_iter, loss, trainer, num_epochs, device):
    all_start_time = time.time()
    num_batches = len(train_iter)
    results = [[] for _ in range(4)]  # save train_loss, train_accuracy, test_loss, test_accuracy
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss_epoch = train_acc_epoch = num_instance_epoch = 0
        for i, (x1, x2, y) in tqdm(enumerate(train_iter), total=num_batches):
            train_loss_sum, train_acc_sum, num_instance = train_epoch(net, x1, x2, y, loss, trainer, device)
            train_loss_epoch += train_loss_sum
            train_acc_epoch += train_acc_sum
            num_instance_epoch += num_instance
        end_time = time.time()
        print('epoch {:d}, train_loss {:.4f}, train accuracy {:.4f}, time cost {:.2f} s'.format(
            epoch + 1, train_loss_epoch / num_instance_epoch, train_acc_epoch / num_instance_epoch, end_time - start_time))
        eval_loss, eval_acc = eval_batch(net, test_iter, loss, device)
        print('eval loss {:.4f}, eval accuracy {:.4f}'.format(eval_loss, eval_acc))
        if (epoch + 1) % 2 == 0:
            results[0].append(train_loss_epoch / num_instance_epoch)
            results[1].append(train_acc_epoch / num_instance_epoch)
            results[2].append(eval_loss)
            results[3].append(eval_acc)
    print('训练结束, 总耗时 %.3f 秒' % (time.time() - all_start_time))
    return results


num_epochs, lr = 10, 1e-5
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.Adam([{'params': params_conv}, {'params': params_fc, 'lr': lr*10}],
                           lr=lr, weight_decay=0.001)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
results = train(net, train_iter, test_iter, loss, trainer, num_epochs, device)

# for x1, x2, y in train_iter:
#     print(x1.shape, x2.shape, y.shape)
#     break

plt.figure()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.axis([0, num_epochs, 0, 1])
plt.plot([i for i in range(2, num_epochs + 1, 2)], results[1], label='train_acc')
plt.plot([i for i in range(2, num_epochs + 1, 2)], results[3], label='test_acc')
plt.legend(loc='best', frameon=False)
plt.grid()
plt.title('Accuracy in training')
plt.show()

pred_result = defaultdict(list)

excel = pd.read_excel(r"D:\Data\STS\sts\影像号与病理分级对应.xlsx", usecols=['影像号', 'source']).astype(str)
hospital_map = {f: l for f, l in zip(excel['影像号'], excel['source'])}
net.eval()
for index, (x1, x2, y) in enumerate(train_set):
    # print(x1.shape, x2.shape, y.shape)
    filename = train_rawdata[index][3]
    patient_id = filename[:7] if 'STS' in filename else filename.split('_')[0]
    hospital = hospital_map[patient_id]
    x1_, x2_ = torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0)
    y_logit = net(x1_, x2_).cpu().tolist()[0]
    y = y.cpu().tolist()
    pred_result['文件'].append(filename)
    pred_result['hospital'].append(hospital)
    pred_result['label'].append(y)
    pred_result['score'].append(y_logit)

df = pd.DataFrame(pred_result)
df.to_excel('output.xlsx', index=False)
