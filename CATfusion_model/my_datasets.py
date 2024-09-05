from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import h5py
import numpy as np


class MyDataSet(Dataset):
    """自定义数据集"""
# 初始化的时候传入patch_feature_filepath和svs_name_list, 这样就要和omics的feature排序是一致的，这样需要对齐操作

    def __init__(self, patch_feature_filepaths: list, patch_survtimes: list, patch_censors: list, omics_features_dict: OrderedDict, transform=None):
        self.patch_feature_filepaths = patch_feature_filepaths
        self.patch_survtimes = patch_survtimes
        self.patch_censors = patch_censors
        self.omics_features_dict = omics_features_dict

    def __len__(self):
        return len(self.patch_feature_filepaths)

    def __getitem__(self, item):
        patch_feature_path = self.patch_feature_filepaths[item]

        with h5py.File(patch_feature_path, 'r') as hf:
            feature_map = np.array(hf['features'][:], dtype=np.float32)
            coords = np.array(hf["coords"][:], dtype=np.int32)
        ranked_coords = list(2*coords[:, 0]+coords[:, 1])

        # 从原来的feature中随机选取，超过100patch的话，就随机不重复的选择100个，如果少于100个patch的话就重复选择100个
        if feature_map.shape[0] >= 100:
            ranked_padding_coords = list(np.random.choice(
                ranked_coords, size=100, replace=False))
        else:
            ranked_padding_coords = list(np.random.choice(
                ranked_coords, size=100, replace=True))

        coords_sort_index = sorted(ranked_padding_coords)
        feature_map = feature_map[[
            ranked_coords.index(ele) for ele in coords_sort_index], :]

        all_omics_feature = []
        for omic in self.omics_features_dict:
            tmp_omic_feature = self.omics_features_dict[omic][item, :, :]
            all_omics_feature.append(tmp_omic_feature)
        all_omics_feature = np.concatenate(all_omics_feature, axis=0)

        survtime = self.patch_survtimes[item]
        censor = self.patch_censors[item]

        return torch.from_numpy(feature_map), torch.from_numpy(all_omics_feature), survtime, censor

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        patch_features, omics_features, survtime, censor = tuple(zip(*batch))

        patch_features = torch.stack(patch_features, dim=0)
        omics_features = torch.stack(omics_features, dim=0)
        survtime = torch.as_tensor(survtime)
        censor = torch.as_tensor(censor)
        return patch_features, omics_features, survtime, censor


# if __name__ == "__main__":
#     patch_feature_filepaths = ["/home/huyongfei/PycharmProjects/PublicCode/CV/WSI/SISH-main/DATA/LATENT/150_merged_files_batch1/batch1_alreadyUsedInSISH/40x/ctranspath/TCGA-ZH-A8Y5-01Z-00-DX1.A7172CB3-474B-455E-8F62-DDE31CAA6783.h5",
#                                "/home/huyongfei/PycharmProjects/PublicCode/CV/WSI/SISH-main/DATA/LATENT/150_merged_files_batch1/batch1_alreadyUsedInSISH/40x/ctranspath/TCGA-ZF-A9RM-01Z-00-DX1.68758A74-2409-48F4-811A-EA42B4F8F942.h5",
#                                "/home/huyongfei/PycharmProjects/PublicCode/CV/WSI/SISH-main/DATA/LATENT/150_merged_files_batch1/batch1_alreadyUsedInSISH/40x/ctranspath/TCGA-YG-AA3O-01Z-00-DX1.0BA2C29A-4D35-478F-9E06-A50ED97B55CC.h5",
#                                "/home/huyongfei/PycharmProjects/PublicCode/CV/WSI/SISH-main/DATA/LATENT/150_merged_files_batch1/batch1_alreadyUsedInSISH/40x/ctranspath/TCGA-XV-AAZW-01Z-00-DX1.26C215F6-0EFA-42D9-A3EF-58466997594B.h5"]
#     # "rna_seq", "mirna_seq", "copynumbervariation", "dnamethylationvariation", "snp"
#     omics_features_dict = OrderedDict()
#     omics_features_dict["rna_seq"] = torch.randn(4, 10, 1024)
#     omics_features_dict["mirna_seq"] = torch.randn(4, 2, 1024)
#     omics_features_dict["dnamethylationvariation"] = torch.randn(4, 8, 1024)
#     # omics_features_dict["CopyNumberVariation"] = torch.randn(4, 27, 1024)
#     omics_features_dict["snp"] = torch.randn(4, 10, 1024)
#     patch_survtimes = [1213, 896, 765, 2132]
#     patch_censors = [1, 1, 0, 1]
#     my_datasets = MyDataSet(patch_feature_filepaths,
#                             patch_survtimes, patch_censors, omics_features_dict)
#     print("my_datasets", len(my_datasets))

#     my_dataloader = torch.utils.data.DataLoader(
#         my_datasets, batch_size=4, shuffle=True)

#     dataloader_iter = iter(my_dataloader)
#     patch_feaure, omics_feature, survtime, censor = dataloader_iter.next()
#     print("patch_feaure", patch_feaure.shape)
#     print("omics_feature", omics_feature.shape)
#     print("survtime", survtime)
#     print("censor", censor)

# patch_feaure torch.Size([4, 100, 768])
# omics_feature torch.Size([4, 30, 1024])
# survtime tensor([ 896, 2132,  765, 1213])
# censor tensor([1, 1, 0, 1])
