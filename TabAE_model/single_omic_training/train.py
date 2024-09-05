
from sklearn.decomposition import PCA
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
os.chdir(sys.path[0])
sys.path.append(
    "/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE")
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models.mae_model import MaskedAutoencoderViT  # NOQA: E402
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
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "weights"), exist_ok=True)

    tb_writer = SummaryWriter(logdir=os.path.join(output_dir, "logs"))

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
                feature_raw_matrix = all_data_dict["matrix"]
                print("feature_raw_matrix", feature_raw_matrix.shape)
                # feature_raw_matrix [n_gene x m_sample] (26984, 22099)
                with open(os.path.join("/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE/single_omic_training/outputCopyNumberVariation", "train_pca_model.pkl"), 'rb') as f:
                    pca = pkl.load(f)

                feature_raw_matrix = np.transpose(pca.transform(
                    feature_raw_matrix.T.values))
                feature_raw_matrix = pd.DataFrame(feature_raw_matrix)

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

        test_dataset_name_index = find_all_ele_in_list(
            datasetnames, args.test_dataset_name)

        feature_raw_matrix_scale_factor = pd.DataFrame({
            "mean": feature_raw_matrix.mean(axis=1),
            "std": feature_raw_matrix.std(axis=1)
        })
        feature_raw_matrix_scale_factor.to_csv(os.path.join(
            args.data_path, omics_list+"_scale_factor.tsv"), sep="\t", header=True, index=True)

        feature_scale_matrix = ((
            feature_raw_matrix.T-feature_raw_matrix_scale_factor["mean"])/feature_raw_matrix_scale_factor["std"]).T

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
    train_dataset = MyDataset(
        train_input_M if "_" not in args.omics_label else all_omics_train_input_M, train_labels)
    test_dataset = MyDataset(
        test_input_M if "_" not in args.omics_label else all_omics_test_input_M, test_labels)
    print(args.omics_label, "train dataset length", len(train_dataset))
    print(args.omics_label, "test dataset length", len(test_dataset))

    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=test_dataset.collate_fn)

    model = MaskedAutoencoderViT(sequence_length=test_input_M.shape[1] if "_" not in args.omics_label else all_omics_test_input_M.shape[1],
                                 embed_dim=1024, depth=4, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf

    def lf(x): return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * \
        (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    statistics = []
    train_loss_list = []
    val_loss_list = []
    best_val_loss = 10000000

    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch)

        scheduler.step()

        # validate
        val_loss = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch)

        tags = ["train_loss", "val_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "weights", "model_checkpoint_best.pth"))

        statistics.append({"epoch": epoch,
                           "train_loss": train_loss,
                           "val_loss": val_loss,
                           "learning_rate": optimizer.param_groups[0]["lr"]})
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(
                output_dir, "weights", "model_checkpoint_epoch{0}.pth".format(epoch)))
    write_pickle(statistics, os.path.join(output_dir, "train_statistics.pkl"))

    N = np.arange(0, args.epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, np.array(train_loss_list), label="train_loss")
    plt.plot(N, np.array(val_loss_list), label="val_loss")
    plt.title("training Loss")
    plt.xlabel("Epochs #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_loss_plot.pdf"))
    print("\n")
    print("Best Model Val loss: {0:.3f}".format(best_val_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--embedding_dim', type=int, default=1024)

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
