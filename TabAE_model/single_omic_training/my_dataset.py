from torchvision import transforms, datasets
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch
from PIL import Image
import os.path
import os
import sys
os.chdir(sys.path[0])


class MyDataset(Dataset):
    """自定义数据集"""

    def __init__(self, data_matrix: np.array, data_labels: np.array):
        self.data_matrix = data_matrix
        self.data_labels = data_labels

    def __len__(self):
        return self.data_matrix.shape[0]

    def __getitem__(self, item):
        ele = self.data_matrix[item]
        ele_tr = torch.from_numpy(ele).float()
        label = self.data_labels[item]

        return ele_tr, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

# if __name__ =="__main__":
#     data_transform = {
#         "train": transforms.Compose([transforms.CenterCrop(168),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.Resize((128,128)),
#                                      transforms.ToTensor(),
#                                      ]),
#         "val": transforms.Compose([transforms.CenterCrop(168),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.Resize((128,128)),
#                                      transforms.ToTensor(),
#                                      ])
#     }

#     # 实例化训练数据集
#     # train_dataset = CelebADataSet(images_dir="/data/projects/huyongfei/gpunode02_wd/projects/datasets/imagedatasets/CelebA/Img/img_align_celeba",
#     #                           images_class_anno="/data/projects/huyongfei/gpunode02_wd/projects/datasets/imagedatasets/CelebA/Anno/identity_CelebA.txt",
#     #                               images_train_val_status="train",
#     #                               images_train_val_status_filepath="/data/projects/huyongfei/gpunode02_wd/projects/datasets/imagedatasets/CelebA/Eval/list_eval_partition.txt",
#     #                           transform=data_transform["train"])

#     # # 实例化验证数据集
#     val_dataset = CelebADataSet(images_dir="/data/projects/huyongfei/gpunode02_wd/projects/datasets/imagedatasets/CelebA/Img/img_align_celeba",
#                               images_class_anno="/data/projects/huyongfei/gpunode02_wd/projects/datasets/imagedatasets/CelebA/Anno/identity_CelebA.txt",
#                                   images_train_val_status="val",
#                                   images_train_val_status_filepath="/data/projects/huyongfei/gpunode02_wd/projects/datasets/imagedatasets/CelebA/Eval/list_eval_partition.txt",
#                               transform=data_transform["val"])

#     # print("train dataset length",len(train_dataset))
#     print("val dataset length",len(val_dataset))

#     batch_size = 16
#     nw = 8
#     print('Using {} dataloader workers every process'.format(nw))
#     # train_loader = torch.utils.data.DataLoader(train_dataset,
#     #                                            batch_size=batch_size,
#     #                                            shuffle=True,
#     #                                            pin_memory=True,
#     #                                            num_workers=nw,
#     #                                            collate_fn=train_dataset.collate_fn)

#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              pin_memory=False,
#                                              num_workers=0,
#                                              collate_fn=val_dataset.collate_fn)

#     train_iter=iter(val_loader)
#     img,labels=train_iter.next()
#     N, C, H, W = img.shape
#     assert N == 16
#     img = torch.permute(img, (1, 0, 2, 3))
#     img = torch.reshape(img, (C, 4, 4 * H, W))
#     img = torch.permute(img, (0, 2, 1, 3))
#     img = torch.reshape(img, (C, 4 * H, 4 * W))
#     img = transforms.ToPILImage()(img)
#     img.save('output/tmp.jpg')
