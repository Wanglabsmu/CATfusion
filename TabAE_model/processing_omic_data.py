import os
import sys
os.chdir(sys.path[0])

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

import argparse
import pickle as pkl

def main(args):
    if not os.path.exists(args.data_dirpath):
        raise ValueError("data directory not exits.")
    
    assert os.path.exists(args.TCGA_datasets_name_filepath), "TCGA datasets name filepath not exits."
    with open(args.TCGA_datasets_name_filepath,'r') as f_in:
        TCGA_datasets_names=[line.strip().replace("\n","") for line in f_in if len(line)>0]
    
    # TCGA_datasets_names.pop(TCGA_datasets_names.index(args.dataset_name_for_test))
    omics_for_fusion_list=args.omics_for_fusion.split("_")

    omics_for_fusion_dict={}
    for omic in omics_for_fusion_list:
        if omic == "RNA-Seq":
            keep_features_num=10240
            omic_data_dirpath=os.path.join(args.data_dirpath,"RNA-Seq","TCGA","all_TCGA_datasets")
            omic_data=pd.read_csv(os.path.join(omic_data_dirpath,TCGA_datasets_names[0]+"_RNAseq_fpkm_matrix.txt"),header=0,sep='\t')
             
            for i in range(1,len(TCGA_datasets_names)):
                tmp=pd.read_csv(os.path.join(omic_data_dirpath,TCGA_datasets_names[i]+"_RNAseq_fpkm_matrix.txt"),header=0,sep='\t')
                tmp.drop("gene_name",axis=1,inplace=True)
                omic_data=pd.merge(omic_data,tmp,on="gene_id")
            
            omic_data.iloc[:,2:].fillna(value=0,method=None,axis=None,inplace=True)
            rowmeans_rank_index=np.argsort(-1*np.mean(omic_data.iloc[:,2:],axis=1))
            omic_data=omic_data.iloc[rowmeans_rank_index,:]
            print(omic,"omic_data",omic_data.shape)
            omic_data.drop_duplicates(subset="gene_name",keep="first",inplace=True)

            omic_data.index=omic_data["gene_name"]
            omic_data.drop("gene_name",axis=1,inplace=True)
            omic_data.drop("gene_id",axis=1,inplace=True)
            
            print(omic,"drop duplicated omic_data",omic_data.shape)
            omic_data_filter_noexpressed=omic_data[np.sum(omic_data>1,axis=1)>2]
            print(omic,"omic no express",omic_data_filter_noexpressed.shape)

            features_cv=omic_data_filter_noexpressed.std(axis=1)/omic_data_filter_noexpressed.mean(axis=1)
            features_cv_rank_index=np.argsort(-1*features_cv)
            omic_data_cv_filtered=omic_data_filter_noexpressed.iloc[features_cv_rank_index[:keep_features_num],:]
            print(omic,"cv filtered",omic_data_cv_filtered.shape)

            sample_name_mapping_filelist=["gdc_sample_sheet_batch1.tsv","gdc_sample_sheet_batch2.tsv"]
            sample_name_mapping=[]
            for filename in sample_name_mapping_filelist:
                tmp_filepath=os.path.join(omic_data_dirpath,filename)
                tmp=pd.read_csv(tmp_filepath,sep="\t",header=0)
                sample_name_mapping.append(tmp)
            sample_name_mapping=pd.concat(sample_name_mapping)
            colnames=omic_data_cv_filtered.columns.tolist()
            colnames_index=[list(sample_name_mapping["File ID"]).index(ele) for ele in colnames]
            samplenames=list(sample_name_mapping["Sample ID"].iloc[colnames_index])
            datasetnames=list(sample_name_mapping["Project ID"].iloc[colnames_index])
            print(omic,len(samplenames)==omic_data_cv_filtered.shape[1],samplenames[:5])

            omic_dict={"fpkm_matrix":omic_data,
                       "filter_no_expression_fpkm_matrix":omic_data_filter_noexpressed,
                       "cv_filtered_fpkm_matrix":omic_data_cv_filtered,
                       "sample_num":omic_data_cv_filtered.shape[1],
                       "feature_num":omic_data_cv_filtered.shape[0],
                       "samplenames":samplenames,
                       "datasetnames":datasetnames}
            
            with open(os.path.join("./datasets/TCGA", omic+'_TCGA_dataset.pkl'), 'wb') as f:
                pkl.dump(omic_dict, f, pkl.HIGHEST_PROTOCOL)
            # omics_for_fusion_dict[omic]=omic_dict

        if omic == "miRNA-Seq":
            omic_data_dirpath=os.path.join(args.data_dirpath,"miRNA-Seq","TCGA")
            omic_data=pd.read_csv(os.path.join(omic_data_dirpath,TCGA_datasets_names[0]+"_miRNAseq_exp_M_reads_per_million_miRNA_mapped.txt"),header=0,sep='\t')
             
            for i in range(1,len(TCGA_datasets_names)):
                tmp=pd.read_csv(os.path.join(omic_data_dirpath,TCGA_datasets_names[i]+"_miRNAseq_exp_M_reads_per_million_miRNA_mapped.txt"),header=0,sep='\t')
                
                omic_data=pd.merge(omic_data,tmp,on="miRNA_ID")
            
            omic_data.iloc[:,1:].fillna(value=0,method=None,axis=None,inplace=True)
            rowmeans_rank_index=np.argsort(-1*np.mean(omic_data.iloc[:,1:],axis=1))
            omic_data=omic_data.iloc[rowmeans_rank_index,:]
            print(omic,"omic_data",omic_data.shape)
            omic_data.drop_duplicates(subset="miRNA_ID",keep="first",inplace=True)

            omic_data.index=omic_data["miRNA_ID"]
            omic_data.drop("miRNA_ID",axis=1,inplace=True)
            
            print(omic,"drop duplicated omic_data",omic_data.shape)
            omic_data_filter_noexpressed=omic_data[np.sum(omic_data>1,axis=1)>2]
            print(omic,"omic no express",omic_data_filter_noexpressed.shape)

            sample_name_mapping_filelist=["gdc_sample_sheet_batch1.tsv","gdc_sample_sheet_batch2.tsv"]
            sample_name_mapping=[]
            for filename in sample_name_mapping_filelist:
                tmp_filepath=os.path.join(omic_data_dirpath,filename)
                tmp=pd.read_csv(tmp_filepath,sep="\t",header=0)
                sample_name_mapping.append(tmp)
            sample_name_mapping=pd.concat(sample_name_mapping)
            colnames=omic_data_filter_noexpressed.columns.tolist()
            colnames_index=[list(sample_name_mapping["File Name"]).index(ele) for ele in colnames]
            samplenames=list(sample_name_mapping["Sample ID"].iloc[colnames_index])
            datasetnames=list(sample_name_mapping["Project ID"].iloc[colnames_index])
            print(omic,len(samplenames)==omic_data_filter_noexpressed.shape[1],samplenames[:5])

            omic_dict={"fpkm_matrix":omic_data,
                       "filter_no_expression_fpkm_matrix":omic_data_filter_noexpressed,
                       "sample_num":omic_data_filter_noexpressed.shape[1],
                       "feature_num":omic_data_filter_noexpressed.shape[0],
                       "samplenames":samplenames,
                       "datasetnames":datasetnames}
            with open(os.path.join("./datasets/TCGA", omic+'_TCGA_dataset.pkl'), 'wb') as f:
                pkl.dump(omic_dict, f, pkl.HIGHEST_PROTOCOL)            
            # omics_for_fusion_dict[omic]=omic_dict 
        
        if omic == "DNAMethylationVariation":
            keep_features_num=20480
            omic_data_dirpath=os.path.join(args.data_dirpath,"DNAMethylationVariation","TCGA")
            omic_data=pd.read_csv(os.path.join(omic_data_dirpath,TCGA_datasets_names[0]+"_DNA_methylation_M.txt"),header=0,sep='\t')
             
            for i in range(1,len(TCGA_datasets_names)):
                tmp=pd.read_csv(os.path.join(omic_data_dirpath,TCGA_datasets_names[i]+"_DNA_methylation_M.txt"),header=0,sep='\t')
                
                omic_data=pd.merge(omic_data,tmp,on="cpg_id")
            
            omic_data.iloc[:,1:].fillna(value=0,method=None,axis=None,inplace=True)
            rowmeans_rank_index=np.argsort(-1*np.mean(omic_data.iloc[:,1:],axis=1))
            omic_data=omic_data.iloc[rowmeans_rank_index,:]
            print(omic,"omic_data",omic_data.shape)
            omic_data.drop_duplicates(subset="cpg_id",keep="first",inplace=True)

            omic_data.index=omic_data["cpg_id"]
            omic_data.drop("cpg_id",axis=1,inplace=True)
            
            print(omic,"drop duplicated omic_data",omic_data.shape)
            
            features_cv=omic_data.std(axis=1)/omic_data.mean(axis=1)
            features_cv_rank_index=np.argsort(-1*features_cv)
            omic_data_cv_filtered=omic_data.iloc[features_cv_rank_index[:keep_features_num],:]
            print(omic,"cv filtered",omic_data_cv_filtered.shape)

            sample_name_mapping_filelist=["gdc_sample_sheet_batch1.tsv","gdc_sample_sheet_batch2.tsv","gdc_sample_sheet_batch3.tsv","gdc_sample_sheet_batch4.tsv","gdc_sample_sheet_batch5.tsv"]
            sample_name_mapping=[]
            for filename in sample_name_mapping_filelist:
                tmp_filepath=os.path.join(omic_data_dirpath,filename)
                tmp=pd.read_csv(tmp_filepath,sep="\t",header=0)
                sample_name_mapping.append(tmp)
            sample_name_mapping=pd.concat(sample_name_mapping)
            colnames=omic_data_cv_filtered.columns.tolist()
            colnames_index=[list(sample_name_mapping["File Name"]).index(ele) for ele in colnames]
            samplenames=list(sample_name_mapping["Sample ID"].iloc[colnames_index])
            datasetnames=list(sample_name_mapping["Project ID"].iloc[colnames_index])
            print(omic,len(samplenames)==omic_data_cv_filtered.shape[1],samplenames[:5])

            omic_dict={"matrix":omic_data,
                       "cv_filtered_matrix":omic_data_cv_filtered,
                       "sample_num":omic_data_cv_filtered.shape[1],
                       "feature_num":omic_data_cv_filtered.shape[0],
                       "samplenames":samplenames,
                       "datasetnames":datasetnames}
            with open(os.path.join("./datasets/TCGA", omic+'_TCGA_dataset.pkl'), 'wb') as f:
                pkl.dump(omic_dict, f, pkl.HIGHEST_PROTOCOL)            
            # omics_for_fusion_dict[omic]=omic_dict

        if omic == "CopyNumberVariation":
            keep_features_num=10240
            omic_data_dirpath=os.path.join(args.data_dirpath,"CopyNumberVariation","TCGA")
            omic_data=pd.read_csv(os.path.join(omic_data_dirpath,TCGA_datasets_names[0]+"_gistic2_final_output_all_data_by_genes_M.tsv"),header=0,sep='\t')
            omic_data.drop("Cytoband",axis=1,inplace=True)

            for i in range(1,len(TCGA_datasets_names)):
                tmp=pd.read_csv(os.path.join(omic_data_dirpath,TCGA_datasets_names[i]+"_gistic2_final_output_all_data_by_genes_M.tsv"),header=0,sep='\t')
                tmp.drop(["Cytoband","Gene Symbol"],axis=1,inplace=True)
                omic_data=pd.merge(omic_data,tmp,on="Gene ID")
          
            omic_data.iloc[:,2:].fillna(value=0,method=None,axis=None,inplace=True)
            print(omic,"omic_data",omic_data.shape)
            features_cv=omic_data.iloc[:,2:].std(axis=1)/omic_data.iloc[:,2:].mean(axis=1)
            features_cv_rank_index=np.argsort(-1*features_cv)
            omic_data=omic_data.iloc[features_cv_rank_index,:]
            omic_data.drop_duplicates(subset="Gene Symbol",keep="first",inplace=True)
            print(omic,"drop duplicated omic_data",omic_data.shape)

            omic_data.index=omic_data["Gene Symbol"]
            omic_data.drop("Gene Symbol",axis=1,inplace=True)
            omic_data.drop("Gene ID",axis=1,inplace=True)
            omic_data=omic_data+1

            omic_data_cv_filtered=omic_data.iloc[:keep_features_num,:]
            print(omic,"cv filtered",omic_data_cv_filtered.shape) 
            print(omic_data_cv_filtered.iloc[:5,:5])

            colnames=omic_data_cv_filtered.columns.tolist()
            samplenames=[ele[:16] for ele in colnames]

            sample_name_mapping_filelist=["gdc_sample_sheet_batch1.tsv","gdc_sample_sheet_batch2.tsv","gdc_sample_sheet_batch3.tsv","gdc_sample_sheet_batch4.tsv","gdc_sample_sheet_batch5.tsv","gdc_sample_sheet_batch6.tsv","gdc_sample_sheet_batch7.tsv","gdc_sample_sheet_batch8.tsv"]
            sample_name_mapping=[]
            for filename in sample_name_mapping_filelist:
                tmp_filepath=os.path.join(omic_data_dirpath,filename)
                tmp=pd.read_csv(tmp_filepath,sep="\t",header=0)
                sample_name_mapping.append(tmp)
            sample_name_mapping=pd.concat(sample_name_mapping)
            samplenames_index=[list(sample_name_mapping["Sample ID"]).index(ele) for ele in samplenames]
            datasetnames=list(sample_name_mapping["Project ID"].iloc[samplenames_index])

            print(omic,len(samplenames)==omic_data_cv_filtered.shape[1],samplenames[:5])

            omic_dict={"matrix":omic_data,
                       "cv_filtered_matrix":omic_data_cv_filtered,
                       "sample_num":omic_data_cv_filtered.shape[1],
                       "feature_num":omic_data_cv_filtered.shape[0],
                       "samplenames":samplenames,
                       "datasetnames":datasetnames}
            with open(os.path.join("./datasets/TCGA", omic+'_TCGA_dataset.pkl'), 'wb') as f:
                pkl.dump(omic_dict, f, pkl.HIGHEST_PROTOCOL)            
            # omics_for_fusion_dict[omic]=omic_dict
        
        if omic == "SNP":
            keep_features_num=10240
            omic_data_dirpath=os.path.join(args.data_dirpath,"SNP","TCGA")
            omic_data=[]
            for i in range(0,len(TCGA_datasets_names)):
                tmp=pd.read_csv(os.path.join(omic_data_dirpath,TCGA_datasets_names[i]+"dataset_all_SNP_M.txt"),header=0,sep='\t')
                tmp["dataset"]=np.repeat(TCGA_datasets_names[i],tmp.shape[0])
                omic_data.append(tmp)
            
            omic_data=pd.concat(omic_data,ignore_index=True)
            print(omic,omic_data["Variant_Type"].value_counts())
            omic_data["count"]=np.repeat(1,omic_data.shape[0])
            omic_data["Gene_Variant_Type"]=[k+"_"+v for k,v in zip(omic_data["Hugo_Symbol"],omic_data["Variant_Type"])]

            unmelted_omic_data = omic_data.pivot_table(index='Gene_Variant_Type', columns='Tumor_Sample_Barcode', values='count',aggfunc='sum')
            
            print(omic,"unmelted_omic_data",unmelted_omic_data.shape)
            unmelted_omic_data.fillna(value=0,method=None,axis=None,inplace=True)


            features_cv=unmelted_omic_data.std(axis=1)/unmelted_omic_data.mean(axis=1)
            features_cv_rank_index=np.argsort(-1*features_cv)
            unmelted_omic_data=unmelted_omic_data.iloc[features_cv_rank_index,:]
            unmelted_omic_data_cv_filtered=unmelted_omic_data.iloc[:keep_features_num,:]

            colnames=unmelted_omic_data.columns.tolist()
            samplenames=[ele[:16] for ele in colnames]

            datasetname_map=omic_data.loc[:,['Tumor_Sample_Barcode','dataset']]
            datasetname_map.drop_duplicates(inplace=True)

            print(omic,len(samplenames)==unmelted_omic_data_cv_filtered.shape[1],samplenames[:5])
            samplenames_index=[list(datasetname_map["Tumor_Sample_Barcode"]).index(ele) for ele in colnames]
            datasetnames=list(datasetname_map["dataset"].iloc[samplenames_index])
            omic_dict={"matrix":unmelted_omic_data,
            "cv_filtered_matrix":unmelted_omic_data_cv_filtered,
            "sample_num":unmelted_omic_data_cv_filtered.shape[1],
            "feature_num":unmelted_omic_data_cv_filtered.shape[0],
            "samplenames":samplenames,
            "datasetnames":datasetnames}
            with open(os.path.join("./datasets/TCGA", omic+'_TCGA_dataset.pkl'), 'wb') as f:
                pkl.dump(omic_dict, f, pkl.HIGHEST_PROTOCOL)            
            # omics_for_fusion_dict[omic]=omic_dict




    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TCGA_datasets_name_filepath', type=str, default="/home/huyongfei/projects/Datasets/all_TCGA_dataset_names.txt")
    parser.add_argument('--dataset_name_for_test', type=str, default="TCGA-TGCT")
    #RNA-Seq_miRNA-Seq_DNAMethylationVariation_CopyNumberVariation_SNP
    parser.add_argument('--omics_for_fusion', type=str, default="RNA-Seq_miRNA-Seq_DNAMethylationVariation_CopyNumberVariation_SNP")
    parser.add_argument('--data_dirpath', type=str, default="/home/huyongfei/projects/Datasets")
    opt = parser.parse_args()
    main(opt)