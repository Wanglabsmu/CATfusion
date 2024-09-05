# **Deep learning-driven survival prediction in pan-cancer studies by integrating multimodal histology-genomic data**

Accurate cancer prognosis is essential for personalized clinical management, guiding treatment strategies and predicting patient survival. Conventional methods, which depend on the subjective evaluation of histopathological features, exhibit significant inter-observer variability and limited predictive power. To overcome these limitations, we developed CATfusion, a deep learning framework that integrates multimodal histology-genomic data for comprehensive cancer survival prediction. By employing self-supervised learning through TabAE for feature extraction and utilizing cross-attention mechanisms to fuse diverse data types, including mRNA-Seq, miRNA-Seq, copy number variation, DNA methylation variation, mutation data and histopathological images. By successfully integrating this multi-tiered patient information, CATfusion has become the first survival prediction model to utilize the most diverse data types across various cancer types. CATfusion achieves superior predictive performance over traditional and unimodal models, as demonstrated by enhanced c-Index and survival AUC scores. The model's high accuracy in stratifying patients into distinct risk groups is a boon for personalized medicine, enabling tailored treatment plans. Moreover, CATfusion's interpretability, enabled by attention-based visualization, offers insights into the biological underpinnings of cancer prognosis, underscoring its potential as a transformative tool in oncology.


### 1. Installing a training environment
```
conda create -n catfusion_1.0 python=3.9 -y
conda activate catfusion_1.0

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardX
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm

pip install transformers==4.18.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scanpy==1.9.1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple igraph==0.10.4
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple python-igraph==0.10.4
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wandb
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple notebook
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scvi-tools
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple datasets
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scib
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchtext==0.12.0 torchdata==0.3.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple louvain
```

### 2. training
```
# single omics data
cd TabAE_model/single_omic_training
python train.py

# wsi data
use AttMIL to extract features from patchs of wsi

# fusion
cd CATfusion_model
python train_five_fold_cross_validation.py
```
