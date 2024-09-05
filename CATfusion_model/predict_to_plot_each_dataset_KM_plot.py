from lifelines.statistics import logrank_test
from pathlib import Path
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
from sksurv.metrics import cumulative_dynamic_auc

import glob
import pandas as pd
import h5py
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from models.model_utils import CIndex_lifeline, cox_log_rank, accuracy_cox, CIndex
from sksurv.datasets import load_veterans_lung_cancer
from lifelines import KaplanMeierFitter
import pickle as pkl
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(32)
torch.manual_seed(32)
torch.cuda.manual_seed_all(32)


def clean_ax(ax, clean_all: bool = False, inverted: bool = False):
    """
    Cleans the borders of a matplotlib.axis object.
    Parameters
    ----------
    ax: matplotlib.axis
        axis object to be modified
    clean_all: boolean (default = False)
        whether to clean the entire boundary
    inverted: boolean (default = False)
        whether to clean bottom and right borders
    """
    if clean_all:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    elif inverted:
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return ax


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.model_weights).parent.parent
    print("output_dir", output_dir)

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

    omics_feature_for_train_dict = OrderedDict()
    for omic in args.omics_names:
        tmp_omic_samplenames = list(
            [ele.decode() for ele in omics_feature_samplename_dict[omic]["samplenames"]])
        tmp_omic_sample_index = [tmp_omic_samplenames.index(
            ele.split("'")[1]) for ele in svs_input_for_model_df[omic+"-selected_samples"]]
        omics_feature_for_train_dict[omic] = omics_feature_samplename_dict[omic]["features"][tmp_omic_sample_index]
        print(omic, "feature size", omics_feature_for_train_dict[omic].shape)

    mydataset = MyDataSet(
        patch_feature_filepaths=list(
            svs_input_for_model_df["patch_feature_filepath"]),
        patch_survtimes=list(svs_input_for_model_df["PFI.time"]),
        patch_censors=list(svs_input_for_model_df["PFI"]),
        omics_features_dict=omics_feature_for_train_dict
    )

    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(mydataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=mydataset.collate_fn)
    print("val_loader", len(val_loader.dataset))

    print("="*72)
    # "rna_seq", "mirna_seq", "copynumbervariation", "dnamethylationvariation", "snp"
    config = CONFIGS["CATfusion"]
    print("config", config)
    # 这里是做生存预测的所以num_classes=1
    model = CATfusion(config, zero_head=True,
                  num_classes=1)
    # 导入训练的权重
    model.load_state_dict(torch.load(args.model_weights,
                          map_location="cpu"), strict=False)

    model = model.to(device)

    model.eval()

    accu_loss = 0.0  # 累计损失

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
    transformer_output_all = []

    with torch.no_grad():
        val_loader = tqdm(val_loader, file=sys.stdout)
        for step, data in enumerate(val_loader):
            patch_feaure, omics_feature, survtime, censor = data

            sample_num += patch_feaure.shape[0]
            patch_feaure = patch_feaure.to(device)
            survtime = (survtime.float()).to(device)
            censor = (censor.float()).to(device)

            omics_input_dict = OrderedDict()
            start_index = 0
            stop_index = 0
            for omic in args.omics_names:
                stop_index = start_index+omics_lens_dict[omic]
                omics_input_dict[omic] = (omics_feature[:,
                                                        start_index:stop_index, :]).to(device)
                start_index = stop_index

            loss, pred, transformer_output = model(
                patch_feaure, omics_input_dict, survtime, censor)

            accu_loss += loss.item()

            risk_pred_all = np.concatenate(
                (risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate(
                (censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate(
                (survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
            transformer_output_all.append(
                transformer_output.detach().cpu().numpy())

    print("accu_loss", accu_loss/sample_num)
    cindex_test = CIndex(risk_pred_all, censor_all, survtime_all)
    cindex_lifeline_test = CIndex_lifeline(
        risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    # pred_test = [risk_pred_all, survtime_all, censor_all]

    print("cindex_test:", cindex_test)
    print("cindex_lifeline_test:", cindex_lifeline_test)
    print("pvalue_test", pvalue_test)
    print("surv_acc_test", surv_acc_test)

    svs_input_for_model_df["pred_hazard"] = risk_pred_all

    from sksurv.nonparametric import kaplan_meier_estimator

    for dataset in np.unique(svs_input_for_model_df["dataset_name"]):
        tmp_svs_input_for_model_df = svs_input_for_model_df[
            svs_input_for_model_df["dataset_name"] == dataset]
        median_risk = np.median(tmp_svs_input_for_model_df['pred_hazard'])
        tmp_svs_input_for_model_df['patient_risk_group'] = [
            "high_risk" if ele > median_risk else "low_risk" for ele in tmp_svs_input_for_model_df['pred_hazard']]

        results = logrank_test(tmp_svs_input_for_model_df['PFI.time'][tmp_svs_input_for_model_df['patient_risk_group'] == 'low_risk'], tmp_svs_input_for_model_df['PFI.time'][tmp_svs_input_for_model_df['patient_risk_group'] == 'high_risk'],
                               event_observed_A=tmp_svs_input_for_model_df['PFI'][tmp_svs_input_for_model_df['patient_risk_group'] == 'low_risk'], event_observed_B=tmp_svs_input_for_model_df['PFI'][tmp_svs_input_for_model_df['patient_risk_group'] == 'high_risk'])

        results.print_summary()

        kmf = KaplanMeierFitter()
        f, ax = plt.subplots(dpi=120, figsize=(5, 4))
        
        kmf.fit(tmp_svs_input_for_model_df['PFI.time'][tmp_svs_input_for_model_df['patient_risk_group'] == 'low_risk'], event_observed=tmp_svs_input_for_model_df['PFI'][tmp_svs_input_for_model_df['patient_risk_group'] == 'low_risk'].astype(
            bool), label="low_risk")
        kmf.plot(ax=ax)
        kmf.fit(tmp_svs_input_for_model_df['PFI.time'][tmp_svs_input_for_model_df['patient_risk_group'] == 'high_risk'], event_observed=tmp_svs_input_for_model_df['PFI'][tmp_svs_input_for_model_df['patient_risk_group'] == 'high_risk'].astype(
            bool), label="high_risk")
        kmf.plot(ax=ax)
        plt.title("Kaplan Meier estimates by predict hazard group")
        plt.xlabel('PFI.time [days] {0:.5f}'.format(results.p_value))
        plt.ylabel("Survival")
        plt.savefig(os.path.join(
            output_dir, "KM_plot_for_each_dataset", '{0}_model_best_weights_pred_hazard_KM_plot.pdf'.format(dataset)))

        # x_standard, y_standard = kaplan_meier_estimator(tmp_svs_input_for_model_df['PFI'][tmp_svs_input_for_model_df['patient_risk_group'] == 'low_risk'].astype(
        #     bool), tmp_svs_input_for_model_df['PFI.time'][tmp_svs_input_for_model_df['patient_risk_group'] == 'low_risk'])
        # x_test, y_test = kaplan_meier_estimator(tmp_svs_input_for_model_df['PFI'][tmp_svs_input_for_model_df['patient_risk_group'] == 'high_risk'].astype(
        #     bool), tmp_svs_input_for_model_df['PFI.time'][tmp_svs_input_for_model_df['patient_risk_group'] == 'high_risk'])

        # print("y_standard", y_standard[:5])
        # f, ax = plt.subplots(dpi=120, figsize=(5, 4))
        # # Standard treatment by age
        # ax.step(x_standard, y_standard, where="post", label='low_risk')
        # ax.step(x_test, y_test, where="post", label='high_risk')
        # ax.set_ylim(0, 1)
        # clean_ax(ax)
        # ax.set_ylabel(r'$\hat{S}_{KM}(t)$')
        # ax.set_xlabel('PFI.time [days] {0:.5f}'.format(results.p_value))
        # ax.set_title(
        #     'Kaplan-Meier estimate by pred hazard + log rank test p-value=')
        # ax.legend(loc='best', frameon=False)
        # plt.savefig(os.path.join(
        #     output_dir, "KM_plot_for_each_dataset", '{0}_model_best_weights_pred_hazard_KM_plot.pdf'.format(dataset)))

        # print("lifelines log rank test p value",
        #       results.p_value)        # 0.7676
        # print("lifelines log rank test test statistic",
        #       results.test_statistic)  # 0.0872


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--patch_feature_dirpath',
                        default='/home/huyongfei/PycharmProjects/PublicCode/CV/WSI/SISH-main/DATA/LATENT', help='/Data/LATENT')
    # "rna_seq", "mirna_seq", "copynumbervariation", "dnamethylationvariation", "snp"
    parser.add_argument('--omics_names_flag', type=str, default="rna_seq-mirna_seq-copynumbervariation-dnamethylationvariation-snp",
                        help='the omics name use to train model')

    parser.add_argument('--model_weights', type=str,
                        default="/home/huyongfei/PycharmProjects/VLP_wsi_RNAseq/wsi_omics_fusion/surivival_preddiction_Progression_free_interval_ChangeModelFC/CATfusion_Progression_free_interval_prediction_changeModelFC/output_rna_seq-mirna_seq-copynumbervariation-dnamethylationvariation-snp_lr1e-05_weight_decay0.7/weights/model_checkpoint_best.pth")

    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    opt.omics_names = opt.omics_names_flag.split(
        "-") if "-" in opt.omics_names_flag else [opt.omics_names_flag]
    main(opt)
