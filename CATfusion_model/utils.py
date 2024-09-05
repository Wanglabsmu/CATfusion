import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from models.model_utils import regularize_weights
from models.model_utils import CIndex_lifeline, cox_log_rank, accuracy_cox


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(
        root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(
        root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key)
                          for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG",
                 ".tif", ".tiff3", ".TIF"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(
        train_images_path) > 0, "number of training images must greater than 0."
    assert len(
        val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def im_convert(tensor):
    """show a pic"""
    image = tensor.cpu().clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image*np.array([0.2612, 0.2547, 0.2572]) + \
        np.array([0.4884, 0.4551, 0.4170])
    image = image.clip(0, 1)

    return image


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 8)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), json_path+" does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    dataiter = iter(data_loader)
    inputs, labels = dataiter.next()

    columns = 4
    rows = 2

    fig = plt.figure(figsize=(20, 12))
    for idx in range(columns*rows):
        ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
        ax.set_title(class_indices[str(labels[idx].item())])
        plt.imshow(im_convert(inputs[idx]))

    plt.savefig("./data_loader_images.jpg")


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, train_omics_names, device, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    omics_lens_dict = {
        "rna_seq": 10,
        "mirna_seq": 2,
        "copynumbervariation": 27,
        "dnamethylationvariation": 8,
        "snp": 10,
    }

    risk_pred_all, censor_all, survtime_all = np.array(
        []), np.array([]), np.array([])

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        patch_feaure, omics_feature, survtime, censor = data

        sample_num += patch_feaure.shape[0]
        patch_feaure = patch_feaure.to(device)
        survtime = (survtime.float()).to(device)
        censor = (censor.float()).to(device)

        omics_input_dict = OrderedDict()
        start_index = 0
        stop_index = 0
        for omic in train_omics_names:
            stop_index = start_index+omics_lens_dict[omic]
            omics_input_dict[omic] = (omics_feature[:,
                                                    start_index:stop_index, :]).to(device)
            start_index = stop_index

        loss_cox, pred, _ = model(
            patch_feaure, omics_input_dict, survtime, censor)
        # loss_reg = regularize_weights(model=model)

        # loss = loss_cox + 0.000002 * loss_reg
        loss = loss_cox
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch,
                                                                  accu_loss.item() / (step + 1)
                                                                  )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        risk_pred_all = np.concatenate(
            (risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate(
            (censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate(
            (survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
        optimizer.step()
        optimizer.zero_grad()
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    print("train epoch {0}".format(epoch), "cindex_test: {0:.4f}, pvalue_test: {1:.4f}, surv_acc_test: {2:.4f}".format(
        cindex_test, pvalue_test, surv_acc_test))
    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, train_omics_names, device, epoch):

    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0

    omics_lens_dict = {
        "rna_seq": 10,
        "mirna_seq": 2,
        "copynumbervariation": 27,
        "dnamethylationvariation": 8,
        "snp": 10,
    }

    risk_pred_all, censor_all, survtime_all = np.array(
        []), np.array([]), np.array([])
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        patch_feaure, omics_feature, survtime, censor = data

        sample_num += patch_feaure.shape[0]
        patch_feaure = patch_feaure.to(device)
        survtime = (survtime.float()).to(device)
        censor = (censor.float()).to(device)

        omics_input_dict = OrderedDict()
        start_index = 0
        stop_index = 0
        for omic in train_omics_names:
            stop_index = start_index+omics_lens_dict[omic]
            omics_input_dict[omic] = (omics_feature[:,
                                                    start_index:stop_index, :]).to(device)
            start_index = stop_index

        loss_cox, pred, _ = model(
            patch_feaure, omics_input_dict, survtime, censor)
        # loss_reg = regularize_weights(model=model)

        # loss = loss_cox + 0.000002 * loss_reg
        loss = loss_cox

        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch,
                                                                  accu_loss.item() / (step + 1))
        risk_pred_all = np.concatenate(
            (risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate(
            (censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate(
            (survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    print("valid epoch {0}".format(epoch), "cindex_test: {0:.4f}, pvalue_test: {1:.4f}, surv_acc_test: {2:.4f}".format(
        cindex_test, pvalue_test, surv_acc_test))
    return accu_loss.item() / (step + 1)
