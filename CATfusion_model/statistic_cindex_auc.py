from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import os
import sys
os.chdir(sys.path[0])


def main(args):

    metrics = []
    for root, dirs, files in os.walk(args.model_predict_res_ditpath):
        for file in files:
            tmp_file = os.path.join(root, file)
            if "model_best_weights_per_cancer_type_criteria.tsv" in tmp_file:
                print(tmp_file)
                fold = tmp_file.split('/')[-2]
                tmp_df = pd.read_csv(tmp_file, header=0, sep="\t", index_col=0)
                tmp_df_colnames = tmp_df.columns.to_list()
                tmp_df.columns = [str(fold)+"-"+ele for ele in tmp_df_colnames]
                metrics.append(tmp_df)

    metrics_df = pd.concat(metrics, axis=1)

    CIndex_M = metrics_df.loc[:, [
        str(i)+"_fold-CIndex_lifeline_value" for i in range(5)]]
    CIndex_M["rowmean"] = CIndex_M.mean(axis=1)
    CIndex_M["std"] = CIndex_M.std(axis=1)
    metrics_df["CIndex_lifeline_value_rowmean"] = CIndex_M["rowmean"]
    metrics_df["CIndex_lifeline_value_std"] = CIndex_M["std"]
    metrics_df.to_csv("CIndex_log_rank_pvalue_stat_res.tsv",
                      header=True, index=True, sep="\t")

    print("ok")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_predict_res_ditpath', type=str,
                        default="/home/huyongfei/PycharmProjects/VLP_wsi_RNAseq/wsi_omics_fusion/surivival_preddiction_Progression_free_interval_ChangeModelFC/CATfusion_Progression_free_interval_prediction_changeModelFC/output_rna_seq-mirna_seq-copynumbervariation-dnamethylationvariation-snp_lr1e-05_weight_decay0.7_fivefold")
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    opt = parser.parse_args()
    main(opt)
