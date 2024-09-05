
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from utils import read_split_data, train_one_epoch, evaluate, write_pickle

from my_dataset import MyDataset
import pickle as pkl
import random
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import argparse
import math
import os
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA

os.chdir(sys.path[0])
sys.path.append(
    "/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE")
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models.mae_model import MaskedAutoencoderViT  # NOQA: E402
from plotUMAP import plot_umap, fit_classifier  # NOQA: E402
# from torch.utils.tensorboard import SummaryWriter


def find_all_ele_in_list(lst, ele):
    name = lst
    first_pos = 0
    res = []
    if ele not in lst:
        return []
    for i in range(name.count(ele)):
        new_list = name[first_pos:]
        next_pos = new_list.index(ele) + 1
        res.append(first_pos + new_list.index(ele))
        first_pos += next_pos
    return res


def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def getData(feature_M: np.array, gap: int = 1024):
    single_cell_list = []
    for single_cell in feature_M:
        feature = []
        length = len(single_cell)
        # spliting the gene expression vector into some sub-vectors whose length is gap
        for k in range(0, length, gap):
            if (k + gap > length):
                a = single_cell[length - gap:length]
            else:
                a = single_cell[k:k + gap]

            # scaling each sub-vectors
            a = preprocessing.scale(
                a, axis=0, with_mean=True, with_std=True, copy=True)
            feature.append(a)

        feature = np.asarray(feature)
        single_cell_list.append(feature)

    single_cell_list = np.asarray(single_cell_list)  # (n_cells,gap_num,gap)
    return single_cell_list


def main(args):
    same_seeds(32)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = "./output"+args.omics_label

    omics_list = args.omics_label.split("_")
    if len(omics_list) > 1:
        # need to concat the omics features
        all_omics_data_dict = {}
        for omic in omics_list:
            data_dict = pkl.load(
                open(os.path.join(args.data_path, omic+"_TCGA_dataset.pkl"), 'rb'))
            if omic == "RNA-Seq":
                feature_raw_matrix = data_dict["cv_filtered_fpkm_matrix"]
            if omic == "miRNA-Seq":
                feature_raw_matrix = data_dict["filter_no_expression_fpkm_matrix"]
            if omic == "DNAMethylationVariation":
                feature_raw_matrix = data_dict["cv_filtered_matrix"]
            if omic == "CopyNumberVariation":
                feature_raw_matrix = np.exp(all_data_dict["matrix"]-1)+1
            if omic == "SNP":
                feature_raw_matrix = data_dict["cv_filtered_matrix"]
            feature_raw_matrix.dropna(inplace=True)

            omic_data_dict = {
                "feature_matrix": feature_raw_matrix,
                "samplenames": data_dict["samplenames"],
                "datasetnames": data_dict["datasetnames"]
            }
            all_omics_data_dict[omic] = omic_data_dict

        overlaped_samplenames = set(data_dict["samplenames"])
        for omic in all_omics_data_dict:
            overlaped_samplenames = overlaped_samplenames.intersection(
                set(all_omics_data_dict[omic]["samplenames"]))

        print("omics overlap sample names length", len(overlaped_samplenames))

        all_omics_train_input_M = []
        all_omics_test_input_M = []
        for omic in all_omics_data_dict:
            omic_data_dict = all_omics_data_dict[omic]
            feature_raw_matrix = omic_data_dict["feature_matrix"]
            samplenames = omic_data_dict["samplenames"]
            datasetnames = omic_data_dict["datasetnames"]
            overlaped_samplenames_index = [samplenames.index(
                ele) for ele in overlaped_samplenames]
            feature_raw_matrix = feature_raw_matrix.iloc[:,
                                                         overlaped_samplenames_index]
            datasetnames = [datasetnames[i] for i in range(
                len(datasetnames)) if i in overlaped_samplenames_index]
            test_dataset_name_index = find_all_ele_in_list(
                datasetnames, args.test_dataset_name)

            feature_raw_matrix_scale_factor = pd.DataFrame({
                "mean": feature_raw_matrix.mean(axis=1),
                "std": feature_raw_matrix.std(axis=1)+0.0000001
            })
            feature_raw_matrix_scale_factor.to_csv(os.path.join(
                args.data_path, args.omics_label+"_of_"+omic+"_scale_factor.tsv"), sep="\t", header=True, index=True)
            feature_scale_matrix = ((
                feature_raw_matrix.T-feature_raw_matrix_scale_factor["mean"])/feature_raw_matrix_scale_factor["std"]).T

            train_feature_df = feature_scale_matrix.iloc[:, [i for i in range(
                feature_scale_matrix.shape[1]) if i not in test_dataset_name_index]]
            test_feature_df = feature_scale_matrix.iloc[:,
                                                        test_dataset_name_index]
            train_feature_M = np.transpose(train_feature_df.values)
            test_feature_M = np.transpose(test_feature_df.values)
            train_input_M = getData(train_feature_M, gap=args.embedding_dim)
            test_input_M = getData(test_feature_M, gap=args.embedding_dim)
            train_labels = np.array([ele for ele in datasetnames if ele !=
                                    args.test_dataset_name])
            test_labels = np.array([ele for ele in datasetnames if ele ==
                                    args.test_dataset_name])
            all_omics_train_input_M.append(train_input_M)
            all_omics_test_input_M.append(test_input_M)

        all_omics_train_input_M = np.concatenate(
            all_omics_train_input_M, axis=1)
        all_omics_test_input_M = np.concatenate(all_omics_test_input_M, axis=1)
        print(args.omics_label, "all_omics_train_input_M size",
              all_omics_train_input_M.shape)
        print(args.omics_label, "all_omics_test_input_M size",
              all_omics_test_input_M.shape)

        le = LabelEncoder()
        le.fit(np.array(datasetnames))
        train_labels = le.transform(train_labels)
        test_labels = le.transform(test_labels)
        print("classes", le.classes_)

    else:
        # don't need to concat the omics features
        # scale the omic feature matrix
        # select test dataset
        omics_list = omics_list[0]
        all_data_dict = pkl.load(
            open(os.path.join(args.data_path, omics_list+"_TCGA_dataset.pkl"), 'rb'))
        if omics_list == "RNA-Seq":
            feature_raw_matrix = all_data_dict["cv_filtered_fpkm_matrix"]
        if omics_list == "miRNA-Seq":
            feature_raw_matrix = all_data_dict["filter_no_expression_fpkm_matrix"]
        if omics_list == "DNAMethylationVariation":
            feature_raw_matrix = all_data_dict["cv_filtered_matrix"]
        if omics_list == "CopyNumberVariation":
            feature_raw_matrix = np.exp(all_data_dict["matrix"]-1)+1

        if omics_list == "SNP":
            feature_raw_matrix = all_data_dict["cv_filtered_matrix"]

        feature_raw_matrix.dropna(inplace=True)
        samplenames = all_data_dict["samplenames"]
        datasetnames = all_data_dict["datasetnames"]

        if omics_list == "CopyNumberVariation":
            sample_name_endfix = np.array(
                [int(ele.split("-")[-1][:-1]) for ele in samplenames])
            cancer_sample_index = np.nonzero(sample_name_endfix < 2)
            # print("cancer_sample_index", list(cancer_sample_index[0]))

        feature_raw_matrix_scale_factor = pd.DataFrame({
            "mean": feature_raw_matrix.mean(axis=1),
            "std": feature_raw_matrix.std(axis=1)
        })
        # feature_raw_matrix_scale_factor.to_csv(os.path.join(
        #     args.data_path, omics_list+"_scale_factor.tsv"), sep="\t", header=True, index=True)

        feature_scale_matrix = ((
            feature_raw_matrix.T-feature_raw_matrix_scale_factor["mean"])/feature_raw_matrix_scale_factor["std"]).T

        if omics_list == "CopyNumberVariation":
            feature_scale_matrix = feature_scale_matrix.iloc[:,
                                                             list(cancer_sample_index[0])]
            datasetnames = [datasetnames[i] for i in range(
                len(datasetnames)) if i not in cancer_sample_index[0]]
            samplenames = [samplenames[i] for i in range(
                len(samplenames)) if i not in cancer_sample_index[0]]

        test_dataset_name_index = find_all_ele_in_list(
            datasetnames, args.test_dataset_name)

        train_feature_df = feature_scale_matrix.iloc[:, [i for i in range(
            feature_scale_matrix.shape[1]) if i not in test_dataset_name_index]]
        test_feature_df = feature_scale_matrix.iloc[:, test_dataset_name_index]
        train_feature_M = np.transpose(train_feature_df.values)
        test_feature_M = np.transpose(test_feature_df.values)
        train_input_M = getData(train_feature_M, gap=args.embedding_dim)
        test_input_M = getData(test_feature_M, gap=args.embedding_dim)
        train_labels = np.array([ele for ele in datasetnames if ele !=
                                 args.test_dataset_name])
        test_labels = np.array([ele for ele in datasetnames if ele ==
                                args.test_dataset_name])

        le = LabelEncoder()
        le.fit(np.array(datasetnames))
        train_labels = le.transform(train_labels)
        test_labels = le.transform(test_labels)
        print("classes", le.classes_)

    # 实例化验证数据集
    if "_" in args.omics_label:
        valid_input_M = np.concatenate(
            [all_omics_train_input_M, all_omics_test_input_M], axis=0)
        valid_labels = np.concatenate([train_labels, test_labels], axis=0)
        print("valid_input_M", valid_input_M.shape)
        print("valid_labels", valid_labels.shape)
    else:
        valid_input_M = np.concatenate([train_input_M, test_input_M], axis=0)
        valid_labels = np.concatenate([train_labels, test_labels], axis=0)
        print("valid_input_M", valid_input_M.shape)
        print("valid_labels", valid_labels.shape)

    valid_dataset = MyDataset(valid_input_M, valid_labels)

    print("valid dataset length", len(valid_dataset))

    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=valid_dataset.collate_fn)

    model = MaskedAutoencoderViT(sequence_length=valid_input_M.shape[1],
                                 embed_dim=1024, depth=4, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)

    if os.path.exists(args.check_point_path):
        check_point = torch.load(args.check_point_path, map_location="cpu")
        model.load_state_dict(check_point, strict=False)

    model = model.to(device)
    model.eval()
    valid_output_embedding = []
    valid_output_labels = []
    valid_loader = tqdm(valid_loader, file=sys.stdout)
    with torch.no_grad():
        for batch in valid_loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            latent, mask, ids_restore = model.forward_encoder(
                imgs, mask_ratio=0.0)
            if args.only_use_cls_token:
                valid_output_embedding.append(
                    latent[:, 0, :].cpu().detach().numpy())
            else:
                valid_output_embedding.append(torch.cat([latent[:, 0, :], torch.mean(
                    latent[:, 1:, :], dim=1)], dim=1).cpu().detach().numpy())
            valid_output_labels.append(labels.cpu().detach().numpy())

    valid_output_embedding = np.vstack(valid_output_embedding)
    valid_output_labels = np.concatenate(valid_output_labels, axis=0)
    valid_output_labels = np.array(
        [le.classes_[ele] for ele in valid_output_labels])
    print("valid_output_embedding", valid_output_embedding.shape)
    if args.only_use_cls_token:
        os.makedirs(os.path.join(
            output_dir, "only_use_cls_token"), exist_ok=True)
        plot_umap(valid_output_embedding, valid_output_labels, output_dir)
        fit_classifier(valid_output_embedding, valid_output_labels, output_dir)
    else:
        plot_umap(valid_output_embedding, valid_output_labels, output_dir)
        fit_classifier(valid_output_embedding, valid_output_labels, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--embedding_dim', type=int, default=1024)
    parser.add_argument('--only_use_cls_token', type=bool, default=False)
    parser.add_argument('--check_point_path', type=str,
                        default="/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE/single_omic_training/outputCopyNumberVariation/weights/model_checkpoint_best.pth")

    # RNA-Seq_miRNA-Seq_DNAMethylationVariation_CopyNumberVariation_SNP
    parser.add_argument('--omics_label', type=str,
                        default="CopyNumberVariation")
    parser.add_argument('--test_dataset_name', type=str, default="TCGA-TGCT")
    parser.add_argument('--data_path', type=str,
                        default="/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE/datasets/TCGA")

    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
