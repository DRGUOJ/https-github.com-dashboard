# Python 3.10.9
# 本程序中基于数据集中的图像和训练过的模型生成CAM激活图
import os
import cv2
import pandas as pd
import torch
import numpy as np
from LoadData import combine_img_and_label, STSDataset
from Network import DualInputNet
from torchvision import transforms

train_rawdata = combine_img_and_label(r"D:\Data\STS\new_data_no_limit\output\test2_data",
                                      'Labelfile/all_test2_label.xlsx')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])
train_set = STSDataset(train_rawdata, transform)

net = DualInputNet()
net.load_state_dict(torch.load('model_state/model.pkl'))
net.eval()
# 线性层的权重, (2, 512)
weight_matrix_1 = net.combination[0].weight.data.cpu().numpy()[:, :512]
weight_matrix_2 = net.combination[0].weight.data.cpu().numpy()[:, 512:]


def CAM_image_norm(m: np.ndarray) -> np.ndarray:
    m = m - np.min(m)
    m_img = m / np.max(m)
    return np.uint8(255 * m_img)


def GetCAM(net, data_tensor, predict):
    x1, x2, _ = data_tensor
    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0)
    # predict = net(x1, x2).argmax(dim=1).item()  # 预测类别
    # 前向传播过程中, resnet最后输出的特征, (1, 512, 7, 7)
    feature_1, feature_2 = net.cnn1(x1).last_hidden_state, net.cnn2(x2).last_hidden_state
    # 变换, (512, 49)
    feature_1, feature_2 = feature_1.data.cpu().numpy().reshape((512, -1)), feature_2.data.cpu().numpy().reshape(
        (512, -1))
    # 全连接层里对应某一类别输出的线性权重, (512, )
    w1, w2 = weight_matrix_1[predict], weight_matrix_2[predict]
    # (49, ) -> (7, 7)
    m1, m2 = np.dot(w1, feature_1), np.dot(w2, feature_2)
    m1, m2 = m1.reshape((7, 7)), m2.reshape((7, 7))
    # 转图像前的标准化
    m1_image, m2_image = CAM_image_norm(m1), CAM_image_norm(m2)
    return m1_image, m2_image


def search_rawdata(file_name):
    for index, (_, _, _, name) in enumerate(train_rawdata):
        if name == file_name:
            return index


selected_data = pd.read_excel('AUC_selection/auc_epoch8/test2_epoch8.xlsx')

for i in range(selected_data.shape[0]):
    file_name, label = selected_data.loc[i, '文件'], selected_data.loc[i, 'label']
    print('generating CAM of %s' % file_name)
    index = search_rawdata(file_name)
    m1_cam, m2_cam = GetCAM(net, train_set[index], label)
    raw_img_path = os.path.join(r"D:\Data\STS\new_data_no_limit\output\test2_data", train_rawdata[index][3])
    name_1, name_2 = os.listdir(raw_img_path)
    if 't1' in name_2:
        name_1, name_2 = name_2, name_1
    # 读取原图像
    img_1_raw, img_2_raw = cv2.imread(os.path.join(raw_img_path, name_1)), cv2.imread(
        os.path.join(raw_img_path, name_2))
    height_1, width_1, _ = img_1_raw.shape
    height_2, width_2, _ = img_2_raw.shape
    heatmap_1 = cv2.applyColorMap(cv2.resize(m1_cam, (width_1, height_1)), cv2.COLORMAP_JET)
    heatmap_2 = cv2.applyColorMap(cv2.resize(m2_cam, (width_2, height_2)), cv2.COLORMAP_JET)
    # 生成, 原图与权重图叠加
    out_1 = heatmap_1 * 0.3 + img_1_raw * 0.5
    out_2 = heatmap_2 * 0.3 + img_2_raw * 0.5
    # 输出保存
    output_path = 'cam_image/test2/' + file_name
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    cv2.imwrite(output_path + '/' + 't1.png', out_1)
    cv2.imwrite(output_path + '/' + 't2.png', out_2)


# m1_cam, m2_cam = GetCAM(net, train_set[0])
# # img_raw = cv2.imread(r"D:\Data\STS\new_data_no_limit\output\train_data\1129637_none\1129637_14_t1.png")
# raw_img_path = os.path.join(r"D:\Data\STS\preprocess\train_data", train_rawdata[0][3])
# name_1, name_2 = os.listdir(raw_img_path)
# if 't1' in name_2:
#     name_1, name_2 = name_2, name_1
# img_1_raw, img_2_raw = cv2.imread(os.path.join(raw_img_path, name_1)), cv2.imread(os.path.join(raw_img_path, name_2))
# height_1, width_1, _ = img_1_raw.shape
# height_2, width_2, _ = img_2_raw.shape
# heatmap_1 = cv2.applyColorMap(cv2.resize(m1_cam, (width_1, height_1)), cv2.COLORMAP_JET)
# heatmap_2 = cv2.applyColorMap(cv2.resize(m2_cam, (width_2, height_2)), cv2.COLORMAP_JET)

# out_1 = heatmap_1 * 0.3 + img_1_raw * 0.5
# out_2 = heatmap_2 * 0.3 + img_2_raw * 0.5
#
# cv2.imwrite('cam_test/1.png', out_1)
# cv2.imwrite('cam_test/2.png', out_2)
