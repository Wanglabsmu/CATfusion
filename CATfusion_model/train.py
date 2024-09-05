import os
import sys
os.chdir(sys.path[0])  # NOQA: E402
sys.path.append(os.getcwd())  # NOQA: E402

import math
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torchvision import transforms
from my_datasets import MyDataSet
from models.modeling_CATfusion import CATfusion, CONFIGS  # NOQA: E402
from utils import read_split_data, train_one_epoch, evaluate, write_pickle

import glob
import pandas as pd
import h5py
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(os.getcwd())
    output_dir = "./output"+"_"+args.omics_names_flag+"_"+"lr" + \
        str(args.lr)+"_"+"weight_decay"+str(args.weight_decay)
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "weights"), exist_ok=True)

    tb_writer = SummaryWriter(logdir=os.path.join(output_dir, "logs"))
    svs_files_omics_files_mapping_dirpath = os.path.join(
        "/home/huyongfei/PycharmProjects/VLP_wsi_RNAseq/wsi_omics_fusion/CATfusion_classification_cancer_type", "output_"+args.omics_names_flag)
    svs_files_omics_files_mapping_files = [
        "svs_input_for_train_dataframe.tsv", "svs_input_for_val_dataframe.tsv"]
    svs_files_omics_files_mapping = []
    for ele in svs_files_omics_files_mapping_files:
        tmp = pd.read_csv(os.path.join(
            svs_files_omics_files_mapping_dirpath, ele), header=0, sep="\t")
        svs_files_omics_files_mapping.append(tmp)
    svs_files_omics_files_mapping = pd.concat(
        svs_files_omics_files_mapping, axis=0)
    print(args.omics_names_flag, "svs files omics files mapping records size:",
          svs_files_omics_files_mapping.shape[0])

    svs_records_merge_annotation = svs_files_omics_files_mapping
    svs_records_merge_annotation["simple_sample_name"] = [
        ele[:12] for ele in svs_records_merge_annotation["sample_name"]]

    # 给每个svs加上PFI(progression_fress_interval)相关数据
    svs_recoreds_pathologic_data_all = pd.read_csv(
        "/home/huyongfei/PycharmProjects/VLP_wsi_RNAseq/WSI_classification_progression_free_interval/TCGA_ctranspath_patch_feature_AttMIL/2018_recommend_progression_free_interval.txt", header=0, sep="\t", index_col="id")
    svs_recoreds_pathologic_data = svs_recoreds_pathologic_data_all.loc[:, [
        "bcr_patient_barcode", "PFI", "PFI.time"]]

    # 对PFI进行筛选，确保有值（包含PFI,PFI.time两个值要同时保证有值）
    svs_recoreds_pathologic_data = svs_recoreds_pathologic_data.dropna(axis=0)

    # ['file_id', 'svs_id', 'barcode', 'size', 'file_status', 'dataset_name', 'sample_name', 'vqvae_feature_path', 'simple_sample_name', 'bcr_patient_barcode', 'PFI', 'PFI.time']
    svs_records_add_PFI_annotation = pd.merge(
        left=svs_records_merge_annotation, right=svs_recoreds_pathologic_data, left_on="simple_sample_name", right_on="bcr_patient_barcode", how="inner")

    omics_feature_path_dict = {
        "rna_seq": "/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE/single_omic_training/outputRNA-Seq/len_feature_result.h5",
        "mirna_seq": "/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE/single_omic_training/outputmiRNA-Seq/len_feature_result.h5",
        "copynumbervariation": "/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE/single_omic_training/outputCopyNumberVariation/len_feature_result.h5",
        "dnamethylationvariation": "/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE/single_omic_training/outputDNAMethylationVariation/len_feature_result.h5",
        "snp": "/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE/single_omic_training/outputSNP/len_feature_result.h5",
    }

    omics_feature_samplename_dict = OrderedDict()
    for omic in args.omics_names:
        with h5py.File(omics_feature_path_dict[omic], 'r') as hf:
            features = np.array(hf['features'][:], dtype=np.float32)
            samplenames = np.array(hf['samplenames'][:])
            # print(samplenames[:5])
            omics_feature_samplename_dict[omic] = {"features": features,
                                                   "samplenames": samplenames}

    # 对齐一下patch feature,omics feature的sample name
    svs_input_for_model_df = svs_records_add_PFI_annotation
    print(args.omics_names_flag, "for train survival predict records size:",
          svs_input_for_model_df.shape)

    dataset_le = LabelEncoder()
    svs_input_for_model_df["dataset_id"] = dataset_le.fit_transform(
        np.array(svs_input_for_model_df["dataset_name"]))
    dataset_mapping = pd.DataFrame({
        "dataset_name": dataset_le.classes_,
        "dataset_id": list(range(len(dataset_le.classes_)))
    })
    dataset_mapping.to_csv(os.path.join(
        output_dir, "dataset_name_id_mapping.csv"), header=True, index=False, sep="\t")

    svs_input_for_train_df, svs_input_for_val_df = train_test_split(
        svs_input_for_model_df, test_size=0.2, shuffle=True, stratify=np.array(svs_input_for_model_df["dataset_name"]))
    svs_input_for_train_df.to_csv(os.path.join(
        output_dir, "svs_input_for_train_dataframe.tsv"), header=True, index=False, sep="\t")
    svs_input_for_val_df.to_csv(os.path.join(
        output_dir, "svs_input_for_val_dataframe.tsv"), header=True, index=False, sep="\t")
    print("train dataset distribution", pd.value_counts(
        svs_input_for_train_df["dataset_name"]))
    print("val dataset distribution", pd.value_counts(
        svs_input_for_val_df["dataset_name"]))

    omics_feature_for_train_dict = OrderedDict()
    for omic in args.omics_names:
        tmp_omic_samplenames = list(
            [ele.decode() for ele in omics_feature_samplename_dict[omic]["samplenames"]])
        tmp_omic_sample_index = [tmp_omic_samplenames.index(
            ele.split("'")[1]) for ele in svs_input_for_train_df[omic+"-selected_samples"]]
        omics_feature_for_train_dict[omic] = omics_feature_samplename_dict[omic]["features"][tmp_omic_sample_index]
        print(omic, "feature size", omics_feature_for_train_dict[omic].shape)

    omics_feature_for_val_dict = OrderedDict()
    for omic in args.omics_names:
        tmp_omic_samplenames = list(
            [ele.decode() for ele in omics_feature_samplename_dict[omic]["samplenames"]])
        tmp_omic_sample_index = [tmp_omic_samplenames.index(
            ele.split("'")[1]) for ele in svs_input_for_val_df[omic+"-selected_samples"]]
        omics_feature_for_val_dict[omic] = omics_feature_samplename_dict[omic]["features"][tmp_omic_sample_index]
        print(omic, "feature size", omics_feature_for_val_dict[omic].shape)

    train_dataset = MyDataSet(
        patch_feature_filepaths=list(
            svs_input_for_train_df["patch_feature_filepath"]),
        patch_survtimes=list(svs_input_for_train_df["PFI.time"]),
        patch_censors=list(svs_input_for_train_df["PFI"]),
        omics_features_dict=omics_feature_for_train_dict
    )

    val_dataset = MyDataSet(
        patch_feature_filepaths=list(
            svs_input_for_val_df["patch_feature_filepath"]),
        patch_survtimes=list(svs_input_for_val_df["PFI.time"]),
        patch_censors=list(svs_input_for_val_df["PFI"]),
        omics_features_dict=omics_feature_for_val_dict
    )

    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    print("train_loader", len(train_loader.dataset))
    print("val_loader", len(val_loader.dataset))

    print("="*72)
    # "rna_seq", "mirna_seq", "copynumbervariation", "dnamethylationvariation", "snp"
    config = CONFIGS["CATfusion"]
    print("config", config)
    # 这里是做生存预测的所以num_classes=1
    model = CATfusion(config, zero_head=True, num_classes=1)
    model = model.to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # optimizer = optim.Adam(pg, lr=args.lr, weight_decay=0.02)
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf

    def lf(x): return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * \
        (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=1.2)

    statistics = []
    train_loss_list = []
    val_loss_list = []
    best_val_loss = 100000

    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     train_omics_names=args.omics_names,
                                     device=device,
                                     epoch=epoch)

        scheduler.step()

        # validate
        val_loss = evaluate(model=model,
                            data_loader=val_loader,
                            train_omics_names=args.omics_names,
                            device=device,
                            epoch=epoch)

        tags = ["train_loss",
                "val_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "weights", "model_checkpoint_best.pth"))

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        statistics.append({"epoch": epoch,
                           "train_loss": train_loss,
                           "val_loss": val_loss,
                           "learning_rate": optimizer.param_groups[0]["lr"]})

        if (epoch+1) % 200 == 0:
            torch.save(model.state_dict(), os.path.join(
                output_dir, "weights", "model_checkpoint_epoch{0}.pth".format(epoch)))
    write_pickle(statistics, os.path.join(output_dir, "train_statistics.pkl"))

    N = np.arange(0, args.epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, np.array(train_loss_list), label="train_loss")
    plt.plot(N, np.array(val_loss_list), label="val_loss")
    plt.title("training loss")
    plt.xlabel("Epochs #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_loss_plot.pdf"))
    print("\n")
    print("Best Model Val Loss: {0:.3f}".format(best_val_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lrf', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--generate_patch_feature_model_name', type=str,
                        default="ctranspath")
    parser.add_argument('--patch_feature_dirpath',
                        default='/home/huyongfei/PycharmProjects/PublicCode/CV/WSI/SISH-main/DATA/LATENT', help='/Data/LATENT')
    # "rna_seq", "mirna_seq", "copynumbervariation", "dnamethylationvariation", "snp"
    parser.add_argument('--omics_names_flag', type=str, default="rna_seq-mirna_seq-copynumbervariation-dnamethylationvariation-snp",
                        help='the omics name use to train model')

    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    opt.omics_names = opt.omics_names_flag.split(
        "-") if "-" in opt.omics_names_flag else [opt.omics_names_flag]
    main(opt)
