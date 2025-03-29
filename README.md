# Stratified Ranking for Dense Passage Retrieval
<a href="https://drive.google.com/file/d/1DJmtBAPNUdKXUdctwWIL1W0ozauF5mjq/view?usp=share_link" target="_blank">Stratified Ranking</a> is a technique for better contrastive learning. This code implements Stratified Ranking for the Dense Passage Retrieval problem and is a fork from <a href="https://github.com/luyug/GC-DPR" target="_blank">GC-DPR</a>. You can run the original GC-DPR code by following the guide in [README-GCDPR.md](./README-GCDPR.md). This guide walks you through how to train a SR-DPR model.

## Table of contents
1. [Setup](#setup)
2. [Data](#data)
    - [Data specification](#data-specification)
    - [Prepare data for training](#bytedataset)
3. [Training](#train-srdpr)
    - [Run training](#run-training)
    - [Parameter descriptions](#parameter-descriptions)
## <a name="setup"></a> Setup
### <a name="prerequisites"></a> Prerequisites
SR-DPR is tested on Ubuntu 20.04, Python 3.8, PyTorch 1.12.1 and Huggingface transformers 3.0.2.

### <a name="environment"></a> Environment setup
```shell
$ python3.8 -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
(.venv)$ pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
(.venv)$ pip install hydra-core tensorflow==2.4.1
(.venv)$ pip install .
```

## <a name="data"></a> Data
### <a name="data-specification"></a> Data specification
Input data should be in `json` or `jsonlines` format. Each data sample has the following structure:
<pre>
{
    "sample_id": Integer,
    "questions": [String],
    "positive_contexts": [
        {
            "title": String,
            "text": String
        }
    ],
    "hardneg_contexts": [
        {
            "title": String,
            "text": String
        }
    ]
}
</pre>

### <a name="bytedataset"></a> Prepare data for training
SR-DPR uses a data format called `ByteDataset` to enable random access to samples in a large dataset that may not fit into RAM. This is especially beneficial in distributed training mode, where each process needs to load an exact same dataset on its own, thus multiplically expands RAM by the number of training processes.

To convert the `jsonlines` format to `ByteDataset` format, run the following command.
```shell
(.venv)$ python -m srdpr.data_helpers.create_dataset \
    --input-jsonl-path /path/to/the/input/jsonlines/data \
    --output-bytedataset-path /path/to/the/output/bytedataset/data
```

**Notes**: The directory `/path/to/the/output/bytedataset/data` must already exist. The above script creates 2 files `data.pkl` and `idxs.pkl` in this directory.

## <a name="train-srdpr"></a> Training
### <a name="run-training"></a> Run training
Run the `train_srdpr.py` script and pass appropriate paramters to train a SR-DPR model. Below is an example.
```shell
(.venv)$ python train_srdpr.py \
    train_mode=pos \
    pipeline.pos.forward_batch_size=8 \
    pipeline.pos.contrastive_size=8 \
    do_lower_case=false \
    max_length=256 \
    pretrained_model_cfg=NlpHUST/vibert4news-base-cased \
    encoder_model_type=hf_bert \
    bytedataset.pos=DATA/vbpl/train/bytedataset \
    log_batch_step=10 \
    learning_rate=2e-5 \
    adam_eps=1e-8 \
    max_grad_norm=2.0 \
    warmup_steps=0 \
    total_updates=1000 \
    save_checkpoint_freq=100 \
    grad_cache=false \
    loss_scale=1.0 \
    output_dir=outputs/vbpl/checkpoints \
    log_dir=outputs/vbpl/logs \
    seed=12345 \
    keep_checkpoint_max=5
```

### <a name="parameter-description"></a> Parameter descriptions
See the below table for detailed parameter descriptions.

**Notes**: Parameters whose description is <code>-</code> are provided for compability with the original GC-DPR code (i.e. `train_dense_encoder.py`). Those paramters are not used in `train_srdpr.py`.

<table style="max-width: 1100px">
  <tr>
    <th colspan="4">Parameter</th>
    <th>Description</th>
  </tr>
  <tr>
    <th colspan="5">Encoder parameters</th>
  </tr>
  <tr>
    <td colspan="4"><code>pretrained_model_cfg</code></td>
    <td>Name of a model on huggingface or path to a local directory containing a model. Currently support BERT family only (RoBERTa is not supported)(<strong>Required</strong>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>encoder_model_type</code></td>
    <td>One of {<code>hf_bert</code>, <code>pytext_bert</code>, <code>fairseq_roberta</code>}. This indicates the library which instantiates the model for training (huggingface, pytext, fairseq). (<strong>Default</strong> to <code>hf_bert</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>pretrained_file</code></td>
    <td>Used only when <code>encoder_model_type</code> is not <code>hf_bert</code>, specify the path to a pretrained model file. (<strong>Default</strong> to <code>None</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>model_file</code></td>
    <td>Path to the previously trained checkpoint, e.g. <code>dpr_biencoder.1000</code>. (<strong>Default</strong> to <code>None</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>projection_dim</code></td>
    <td>Dimension of the pooled output of an input sequence. If zero, the embedding vector of <code>[CLS]</code> token is used as the pooled output. Otherwise, the embedding vector of <code>[CLS]</code> is projected by a <code>torch.nn.Linear</code> layer into a <code>project_dim</code>-dimensional vector as the final pooled output. (<strong>Default</strong> to <code>0</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>sequence_length</code></td>
    <td>-</td>
  </tr>
  <tr>
    <th colspan="5">Training parameters</td>
  </tr>
  <tr>
    <td colspan="4"><code>train_file</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>dev_file</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>batch_size</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>dev_batch_size</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>seed</code></td>
    <td>Set this parameter to a same value between runs to ensure reproducible results. (<strong>Default</strong> to <code>12345</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>adam_eps</code></td>
    <td>Parameter of AdamW optimizer (<strong>Default</strong> to <code>1e-8</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>adam_betas</code></td>
    <td>Parameter of AdamW optimizer (<strong>Default</strong> to <code>(0.9, 0.999)</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>max_grad_norm</code></td>
    <td>Globally clip gradients by this value. See <a href="http://proceedings.mlr.press/v28/pascanu13.pdf" target="_blank">http://proceedings.mlr.press/v28/pascanu13.pdf</a> for details. (<strong>Default</strong> to <code>1.0</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>log_batch_step</code></td>
    <td>Logging training loss to stdout every <code>log_batch_step</code> (<strong>Default</strong> to <code>10</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>train_rolling_loss_step</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>weight_decay</code></td>
    <td>Parameter of AdamW optimizer (<strong>Default</strong> to <code>0.0</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>learning_rate</code></td>
    <td>Learning rate of the AdamW optimizer (<strong>Default</strong> to <code>1e-5</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>warmup_steps</code></td>
    <td>Number of warmup steps (<strong>Default</strong> to <code>1000</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>dropout</code></td>
    <td>Dropout rate used for the encoder model (i.e. <code>BertModel</code>) (<strong>Default</code> to <code>0.1)</code></td>
  </tr>
  <tr>
    <td colspan="4"><code>gradient_accumulation_steps</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>num_train_epochs</code></td>
    <td>Number of training epochs (<strong>Default</strong> to <code>3</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>grad_cache</code></td>
    <td>Whether or not to use Gradient Cache. See <a href="https://aclanthology.org/2021.repl4nlp-1.31.pdf" target="_blank">https://aclanthology.org/2021.repl4nlp-1.31.pdf</a> for details (<strong>Default</strong> to <code>None</code>, meaning that Gradient Cache is not used)</td>
  </tr>
  <tr>
    <td colspan="4"><code>q_chunk_size</code></td>
    <td>A batch of queries is splitted into multiple sub-batches, each has <code>q_chunk_size</code> queries, used only when <code>grad_cache</code> is <code>True</code> (<strong>Default</strong> to <code>2</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>ctx_chunk_size</code></td>
    <td>A batch of contexts is splitted into multiple sub-batches, each has <code>ctx_chunk_size</code> contexts, used only when <code>grad_cache</code> is <code>True</code> (<strong>Default</strong> to <code>2</code>)</td>
  </tr>
  <tr>
    <th colspan="5">CUDA parameters</th>
  </tr>
  <tr>
    <td colspan="4"><code>no_cuda</code></td>
    <td>Whether or not to use CUDA (<strong>Default</strong> to <code>None</code>, meaning that CUDA is used)</td>
  </tr>
  <tr>
    <td colspan="4"><code>local_rank</code></td>
    <td>Automatically passed by <code>torch.distributed.launch</code> or <code>-1</code> when running normally (not in distributed mode). If you have <code>3</code> GPU cards and launch <code>3</code> processed for training, those processes will be called with <code>local_rank</code> set to <code>0</code>, <code>1</code> and <code>2</code>. (<strong>Default</strong> to <code>-1</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>fp16</code></td>
    <td>Whether or not to use <code>fp16</code> (<strong>Default</strong> to <code>None</code>, meaning <code>fp16</code> is not used)</td>
  </tr>
  <tr>
    <td colspan="4"><code>fp16_opt_level</code></td>
    <td><code>fp16</code> optimization level. See <a href="https://nvidia.github.io/apex/amp.html" target="_blank">https://nvidia.github.io/apex/amp.html</a> for details. (<strong>Default</strong> to <code>O1</code>), allowed values are <code>O0</code>, <code>O1</code>, <code>O2</code>, <code>O3</code>)</td>
  </tr>
  <tr>
    <th colspan="5">Tokenizer parameters</th>
  </tr>
  <tr>
    <td colspan="4"><code>do_lower_case</code></td>
    <td>This parameter is passed to, for example, the method <code>BertTokenizer.from_pretrained</code> in <code>transformers</code> library. The behavior of this parameter depends on the tokenizer class that you use, e.g. <code>BertTokenizer</code>, <code>PhobertTokenizer</code>, .etc.<br><strong>Note:</strong> for <code>transformers==3.0.2</code>, using <code>BertTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased", do_lower_case=True)</code> strips all accents, leads to wrong tokenization results.<br>(<strong>Default</strong> to <code>False</code>)</td>
  </tr>
  <tr>
    <th colspan="5">Main parameters</th>
  </tr>
  <tr>
    <td colspan="4"><code>eval_per_epoch</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>global_loss_buf_sz</code></td>
    <td>Max size in bytes of data serialized by <code>pickle.dump</code>, i.e. data that needs to be gathered across multiple processes when training in distributed mode, such as embedding vectors of queries and contexts. (<strong>Default</strong> to <code>150000</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>fix_ctx_encoder</code></td>
    <td>Whether or not to freeze weights of the context encoder and train only the query encoder. (<strong>Default</strong> to <code>None</code>, meaning that both the query encoder and context encoder are updated)</td>
  </tr>
  <tr>
    <td colspan="4"><code>output_dir</code></td>
    <td>Path to the directory containing saved checkpoints, i.e. the directory where a checkpoint named <code>dpr_biencoder.1000</code> locates. (<strong>Required</strong>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>log_dir</code></td>
    <td>Path to the directory containing tensorboard logs (<strong>Default</strong> to <code>logs</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>hard_negatives</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>other_negatives</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>train_files_unsample_rates</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>val_av_rank_start_epoch</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>val_av_rank_hard_neg</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>val_av_rank_other_neg</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>val_av_rank_bsz</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>val_av_rank_max_qs</code></td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="4"><code>checkpoint_file_name</code></td>
    <td>Name prefix of checkpoints to be saved (<strong>Default</strong> to <code>dpr_biencoder</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>loss_scale</code></td>
    <td>This number is divided to the loss before backward propagation, thus scale gradients of model weights accordingly, e.g. if <code>loss_scale=8.0</code>, gradients of model weights will be divided by <code>8.0</code> (<strong>Default</strong> to <code>8.0</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>sim_func</code></td>
    <td>The similarity function used to calculate similarity score between a query embedding vector and a context embedding vector. (<strong>Default</strong> to <code>dot_product</code>, allowed values are <code>dot_product</code>, <code>cosine</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>keep_checkpoint_max</code></td>
    <td>Maximum number of checkpoints to keep (<strong>Default</strong> to <code>5</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>total_updates</code></td>
    <td>Total optimization steps (<strong>Default</strong> to <code>10000</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>save_checkpoint_freq</code></td>
    <td>Save checkpoint every <code>save_checkpoint_freq</code> optimization steps (<strong>Default</strong> to <code>1000</code>)</td>
  </tr>
  <tr>
    <th colspan="5">Pipeline parameters</th>
  </tr>
  <tr>
    <td colspan="4"><code>regulate_factor</code></td>
    <td>Suppose <code>regulate_factor=3</code> and <code>train_mode=pos+poshard+hard</code>. During training, it will take <code>3</code> batches from <code>pos</code> pipeline then <code>1</code> batch from <code>poshard</code> pipeline then 1 batch from <code>hard</code> pipeline, repeatedly. Set this to a number larger than 1 in case not all samples in the training dataset has hard negatives attached. See the <a href="https://drive.google.com/file/d/1DJmtBAPNUdKXUdctwWIL1W0ozauF5mjq/view?usp=share_link" target="_blank">SR-DPR paper</a> for details. (<strong>Default</strong> to <code>1</code>)</td>
  </tr>
  <tr>
    <td colspan="4"><code>train_mode</code></td>
    <td>Specify which pipelines to be included during training. Separate each pipeline by <code>+</code>. Available pipelines are <code>inbatch</code>, <code>pos</code>, <code>poshard</code> and <code>hard</code>. (<strong>Default</strong> to <code>poshard+hard</code>)</td>
  </tr>
  <tr>
    <td colspan="2" rowspan="4"><code>bytedataset</code></td>
    <td colspan="2"><code>pos</code></td>
    <td>Path to the <code>pos</code> bytedataset (<strong>Required</strong> if <code>train_mode</code> contains <code>pos</code>)</td>
  </tr>
  <tr>
    <td colspan="2"><code>poshard</code></td>
    <td>Path to the <code>poshard</code> bytedataset (<strong>Required</strong> if <code>train_mode</code> contains <code>poshard</code>)</td>
  </tr>
  <tr>
    <td rowspan="2"><code>hard</code></td>
    <td><code>hardneg</code></td>
    <td>Path to the <code>hardneg</code> bytedataset (<strong>Required</strong> if <code>train_mode</code> contains <code>hard</code>)</td>
  </tr>
  <tr>
    <td><code>randneg</code></td>
    <td>Path to the <code>randneg</code> bytedataset (<strong>Required</strong> if <code>train_mode</code> contains <code>hard</code> and <code>pipeline.hard.use_randneg</code> is <code>True</code>)</td>
  </tr>
  <tr>
    <td rowspan="18"><code>pipeline</code></td>
    <td rowspan="5" colspan="2"><code>inbatch</code></td>
    <td><code>forward_batch_size</code></td>
    <td>Batch size of <code>inbatch</code> pipeline (<strong>Default</strong> to <code>16</code>)</td>
  </tr>
  <tr>
    <td><code>use_hardneg</code></td>
    <td>Whether or not to hard negatives for <code>inbatch</code> pipeline (<strong>Default</strong> to <code>True</code>)</td>
  </tr>
  <tr>
    <td><code>use_num_hardnegs</code></td>
    <td>Number of hard negatives to be used for each training step. In case the actual number of hard negatives is larger than <code>use_num_hardnegs</code>, do sampling. E.g. if the actual number of hard negatives is <code>7</code> and <code>use_num_hardnegs</code> is <code>5</code>, then sample <code>5</code> out of <code>7</code> actual hard negatives. (<strong>Default</strong> to <code>1</code>)</td>
  </tr>
  <tr>
    <td><code>gradient_accumulate_steps</code></td>
    <td>Number of backwards per an optimizer update for <code>inbatch</code> pipeline, i.e. gradients are accumulated after each backward. (<strong>Default</strong> to <code>1</code>)</td>
  </tr>
  <tr>
    <td><code>logging_steps</code></td>
    <td>Logging <code>inbatch</code> loss to tensorboard every <code>logging_steps</code> (<strong>Default</strong> to <code>10</code>)</td>
  </tr>
  <tr>
    <td rowspan="4" colspan="2"><code>pos</code></td>
    <td><code>forward_batch_size</code></td>
    <td>Batch size of <code>pos</code> pipeline (<strong>Default</strong> to <code>16</code>)</td>
  </tr>
  <tr>
    <td><code>contrastive_size</code></td>
    <td>Number of negatives used in N-pair loss for <code>pos</code> pipeline (<strong>Default</strong> to <code>16</code>)</td>
  </tr>
  <tr>
    <td><code>gradient_accumulate_steps</code></td>
    <td>Number of backwards per an optimizer update for <code>pos</code> pipeline, i.e. gradients are accumulated after each backward. (<strong>Default</strong> to <code>1</code>)</td>
  </tr>
  <tr>
    <td><code>logging_steps</code></td>
    <td>Logging <code>pos</code> loss to tensorboard every <code>logging_steps</code> (<strong>Default</strong> to <code>10</code>)</td>
  </tr>
  <tr>
    <td rowspan="4" colspan="2" style="background-color: white;"><code>poshard</code></td>
    <td><code>forward_batch_size</code></td>
    <td>Batch size of <code>poshard</code> pipeline (<strong>Default</strong> to <code>2</code>)</td>
  </tr>
  <tr>
    <td><code>constrastive_size</code></td>
    <td>Number of negatives used in N-pair loss for <code>poshard</code> pipeline (<strong>Default</strong> to <code>2</code>)</td>
  </tr>
  <tr>
    <td><code>gradient_accumulate_steps</code></td>
    <td>Number of backwards per an optimizer update for <code>poshard</code> pipeline, i.e. gradients are accumulated after each backward. (<strong>Default</strong> to <code>1</code>)</td>
  </tr>
  <tr>
    <td><code>logging_steps</code></td>
    <td>Logging <code>poshard</code> loss to tensorboard every <code>logging_steps</code> (<strong>Default</strong> to <code>10</code>)</td>
  </tr>
  <tr>
    <td rowspan="5" colspan="2"><code>hard</code></td>
    <td><code>forward_batch_size</code></td>
    <td>Batch size of <code>hard</code> pipeline (<strong>Default</strong> to <code>2</code>)</td>
  </tr>
  <tr>
    <td><code>contrastive_size</code></td>
    <td>Number of negatives used in N-pair loss for <code>hard</code> pipeline (<strong>Default</strong> to <code>2</code>)</td>
  </tr>
  <tr>
    <td><code>gradient_accumulate_steps</code></td>
    <td>Number of backwards per an optimizer update for <code>hard</code> pipeline, i.e. gradients are accumulated after each backward. (<strong>Default</strong> to <code>1</code>)</td>
  </tr>
  <tr>
    <td><code>logging_steps</code></td>
    <td>Logging <code>hard</code> loss to tensorboard every <code>logging_steps</code> (<strong>Default</strong> to <code>10</code>)</td>
  </tr>
  <tr>
    <td><code>use_randneg</code></td>
    <td>Whether or not to use additional samples without hard negatives in <code>hard</code> pipeline (<strong>Default</strong> to <code>True</code>)</td>
  </tr>
</table>

## References
If you find `SR-DPR` useful, please cite [our paper](https://doi.org/10.1145/3628797.3628810) as follows:
```
@inproceedings{10.1145/3628797.3628810,
    author = {Le Vu, Loi and Nguyen Tien, Dong and Dang Minh, Tuan},
    title = {Stratified Ranking for Dense Passage Retrieval},
    year = {2023},
    isbn = {9798400708916},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3628797.3628810},
    doi = {10.1145/3628797.3628810},
    abstract = {Dense passage retrieval has recently boosted the performance of involved systems such as question answering or search engine. On this problem, prior works trained a dense retriever by learning to rank, i.e. ranking relevant/positive passages higher than irrelevant/negative ones. In this paper, we propose a Stratified Ranking approach for Dense Passage Retrieval (SR-DPR), which performs three-way ranking instead of the typical two-way positive-negative ranking. SR-DPR is concerned with three relevance levels, which are positive, hard negative and random/easy negative. We train SR-DPR model by minimizing a contrastive negative log-likelihood (NLL) loss involved with these three relevance levels, which is a finer-grained version of the N-pair loss [16]. To efficiently implement SR-DPR, we designed three data pipelines, each pipeline is used to learn the contrast between two out of three relevance levels. SR-DPR outperforms the strong baseline DPR [5] by 0.6-1.5\% retrieval accuracy on Natural questions [6] dataset and 3-6\% of that on Zalo Legal Text Retrieval dataset. SR-DPR also gives competitive results compared with current state-of-the-art methods without requiring complicated training regime or intensive hardware resources. The idea of stratified ranking of SR-DPR is not restricted to the scope of dense passage retrieval but can be applied in any contrastive learning problem. We conducted detailed ablation studies to give insights into SR-DPR’s behavior.},
    booktitle = {Proceedings of the 12th International Symposium on Information and Communication Technology},
    pages = {24–31},
    numpages = {8},
    keywords = {Contrastive learning, Data pipeline, Dense passage retrieval, Stratified ranking},
    location = {Ho Chi Minh, Vietnam},
    series = {SOICT '23}
}
```
Also consider citing the original studies this work builds on:
* [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
    ```
    @inproceedings{karpukhin-etal-2020-dense,
        title = "Dense Passage Retrieval for Open-Domain Question Answering",
        author = "Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau",
        booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        month = nov,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2020.emnlp-main.550",
        doi = "10.18653/v1/2020.emnlp-main.550",
        pages = "6769--6781",
    }
    ```
* [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983)
    ```
    @inproceedings{gao2021scaling,
         title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
         author={Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan},
         booktitle ={Proceedings of the 6th Workshop on Representation Learning for NLP},
         year={2021},
    }
    ```
