# TIMIT ASR with Transformer/Conformer (Streaming/Non-Streaming)

This example contains code used to train a Transformer or Conformer model (both Streaming and Non-Streaming versions) with the [TIMIT dataset](https://www.nist.gov/sites/default/files/documents/timit_acd-14.3.31.tar.gz).

## Overview

All the scripts you need are in the `run.sh`. There are several stages in the `run.sh`, and each stage has its function. Below is a detailed description of each stage, as well as instructions on how to use the script.

| Stage | Function |
|:------|:----------------------------------------------------------- |
| 0     | Process data. It includes: <br> (1) Download/Prepare the dataset <br> (2) Calculate the CMVN (if necessary) <br> (3) Get the vocabulary file <br> (4) Prepare manifest files for train, development, and test datasets |
| 1     | Train the model |
| 2     | Get the final model by averaging the top-k models. Setting k=1 means choosing the best model. |
| 3     | Test the final model's performance |
| 4     | (Optional) Perform CTC alignment on test data |
| 5     | (Optional) Export the model checkpoint for production use |

You can choose to run a range of stages by setting the `stage` and `stop_stage` variables. For example, if you want to execute the code in stages 2 and 3, you can run:

```bash
bash run.sh --stage 2 --stop_stage 3
```

Or you can set `stage` equal to `stop_stage` to only run one stage. For example, if you only want to run `stage 0`, you can use:

```bash
bash run.sh --stage 0 --stop_stage 0
```

## The Environment Variables

The `path.sh` script contains the necessary environment variables. You need to run this script first:

```bash
. path.sh || exit 1;
```

Another script is also required:

```bash
. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;
```

This supports the use of `--variable value` in the shell scripts.

## The Local Variables

Some local variables are set in `run.sh`.

- `gpus`: Denotes the GPU numbers you want to use. If you set `gpus=`, it means you only use CPU.
- `stage`: Denotes the number of the stage you want to start from in the experiments.
- `stop_stage`: Denotes the number of the stage you want to end at in the experiments.
- `conf_path`: Denotes the config path of the model.
- `decode_conf_path`: Configuration path for decoding.
- `avg_num`: Denotes the number k of top-k models you want to average to get the final model.
- `TIMIT_path`: Path to the TIMIT dataset.
- `ckpt`: Denotes the checkpoint prefix of the model (derived from `conf_path`).

You can set these local variables (except `ckpt`) when you use `run.sh`. For example:

```bash
bash run.sh --gpus 0,1 --avg_num 5
```

## Stage 0: Data Processing

Download the TIMIT dataset and extract it to `~/datasets`. Then the dataset is in the directory `~/datasets/TIMIT`.

To use this example, you need to process the data first. You can use stage 0 in `run.sh` to do this. The code is:

```bash
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
 # prepare data
 bash ./local/timit_data_prep.sh ${TIMIT_path}
 bash ./local/data.sh || exit -1
fi
```

If you only want to process the data, you can run:

```bash
bash run.sh --stage 0 --stop_stage 0
```

After processing, the `data` directory will contain the necessary manifest files and other metadata.

## Stage 1: Model Training

To train the model, use stage 1 in `run.sh`. The code is:

```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
 # train model, all `ckpt` under `exp` dir
 CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt}
fi
```

If you want to train the model, you can use:

```bash
bash run.sh --stage 0 --stop_stage 1
```

Or run these commands directly (assuming you have the necessary environment set up):

```bash
. path.sh
bash ./local/timit_data_prep.sh /path/to/TIMIT
bash ./local/data.sh
CUDA_VISIBLE_DEVICES=0 ./local/train.sh conf/transformer.yaml transformer
```

(Note: Replace `/path/to/TIMIT` with the actual path to your TIMIT dataset.)

## Stage 2: Top-k Models Averaging

After training, get the final model for testing and inference by averaging the top-k models. The code is:

```bash
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
 # avg n best model
 avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi
```

To get the final model, use:

```bash
bash run.sh --stage 0 --stop_stage 2
```

Or run these commands directly:

```bash
. path.sh
bash ./local/timit_data_prep.sh /path/to/TIMIT
bash ./local/data.sh
CUDA_VISIBLE_DEVICES=0 ./local/train.sh conf/transformer.yaml transformer
avg.sh best exp/transformer/checkpoints 10
```

## Stage 3: Model Testing

Evaluate the model's performance. The code is:

```bash
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
 # test ckpt avg_n
 CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
```

To train and test the model, use:

```bash
bash run.sh --stage 0 --stop_stage 3
```

Or run these commands directly:

```bash
. path.sh
bash ./local/timit_data_prep.sh /path/to/TIMIT
bash ./local/data.sh
CUDA_VISIBLE_DEVICES=0 ./local/train.sh conf/transformer.yaml transformer
avg.sh best exp/transformer/checkpoints 10
CUDA_VISIBLE_DEVICES=0 ./local/test.sh conf/transformer.yaml conf/tuning/decode.yaml exp/transformer/checkpoints/avg_10
```

## Stage 4: (Optional) CTC Alignment

Perform CTC alignment on test data. The code is:

```bash
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
 # ctc alignment of test data
 CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
```

To run CTC alignment, use:

```bash
bash run.sh --stage 0 --stop_stage 4
```

Or run the necessary commands directly.

## Stage 5: (Optional) Model Export

Export the model checkpoint for production use. **Note: The stage number `5` is a placeholder and may need to be adjusted based on your specific use case.** The code is:

```bash
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
 # export ckpt avg_n
 CUDA_VISIBLE_DEVICES= ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit
fi
```

To export the model, use:

```bash
bash run.sh --stage 0 --stop_stage 5
```

Or run the export command directly.
