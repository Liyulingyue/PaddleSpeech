# Transformer/Conformer ASR with Librispeech
This example contains code used to train [u2](https://arxiv.org/pdf/2012.05481.pdf) model (Transformer or [Conformer](https://arxiv.org/pdf/2005.08100.pdf) model) with [Librispeech dataset](http://www.openslr.org/resources/12)

## Overview
All the scripts you need are in the `run.sh`. There are several stages in the `run.sh`, and each stage has its function.

| Stage | Function                                                     |
|:---- |:----------------------------------------------------------- |
| 0     | Process data. It includes: <br>       (1) Download the dataset <br>       (2) Calculate the CMVN of the train dataset <br>       (3) Get the vocabulary file <br>       (4) Get the manifest files of the train, development, and test datasets <br>       (5) Get the sentencepiece model |
| 1     | Train the model                                              |
| 2     | Get the final model by averaging the top-k models, setting k = 1 means choosing the best model                           |
| 3     | Test the final model performance                             |
| 4     | Get CTC alignment of test data using the final model          |
| 5     | Infer a single audio file                                    |
| 51    | Export the final model to a static graph format for deployment |

You can choose to run a range of stages by setting the `stage` and `stop_stage` parameters. 

For example, if you want to execute the code in stage 2 and stage 3, you can run this script:
```bash
bash run.sh --stage 2 --stop_stage 3
```
Or you can set `stage` equal to `stop_stage` to only run one stage.
For example, if you only want to run `stage 0`, you can use the script below:
```bash
bash run.sh --stage 0 --stop_stage 0
```
The script `run.sh` utilizes configuration files, GPU resources, and local scripts to perform the tasks outlined in each stage. Specifically:
- Configuration files are loaded from `conf/transformer.yaml` and `conf/tuning/decode.yaml`.
- GPU devices are specified via the `gpus` variable (e.g., `gpus=0,1,2,3`).
- Local scripts (e.g., `data.sh`, `train.sh`, `avg.sh`, `test.sh`, `align.sh`, `test_wav.sh`, and `export.sh`) handle the respective tasks for data preparation, model training, averaging, testing, alignment, single audio file inference, and model export.

The document below will describe the scripts in the `run.sh` in detail.
## The Environment Variables
The `path.sh` contains the essential environment variables required for the scripts to run correctly. 
```bash
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
```
These scripts need to be sourced first to ensure that all necessary environment variables are set. 

Additionally, another script is also required:
```bash
. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;
```
This script supports the use of `--variable value` options in the shell scripts, allowing for flexible configuration without directly modifying the scripts.

The environment variables set in `path.sh` and `cmd.sh` include paths to directories, executable files, and other configurations necessary for the system to function properly. For example, `MAIN_ROOT` is typically set in `path.sh` to point to the main directory of the project.

The script also uses several other environment variables that are set either directly in the script or through command-line arguments parsed by `parse_options.sh`. These variables include:

- `gpus`: A comma-separated list of GPU IDs to use for training and inference.
- `stage` and `stop_stage`: These variables control which stages of the process to run. For instance, setting `stage=1` and `stop_stage=3` will run only stages 1, 2, and 3.
- `conf_path`: The path to the configuration file for the model.
- `ips`: A placeholder for a comma-separated list of IP addresses (typically used for distributed training).
- `decode_conf_path`: The path to the decoding configuration file.
- `avg_num`: The number of best models to average for improving model performance.
- `audio_file`: The path to an audio file for single-file testing.

These variables are used throughout the script to control the behavior of different stages, such as data preparation, model training, model averaging, testing, alignment, and exporting the model. For example:

```bash
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt} ${ips}
fi

# ... other stages follow similar patterns ...
```

By setting these environment variables appropriately, users can customize the behavior of the script to suit their specific needs and resources.
## The Local Variables

Some local variables are set in the `run.sh` script and can be configured to customize the execution of the script. Here is a detailed explanation of each variable:

`gpus` denotes the GPU number or numbers you want to use for training or inference. If you set `gpus=` (an empty value), it means you only use the CPU. Multiple GPUs can be specified using a comma-separated list, e.g., `0,1,2,3`.

`stage` denotes the number of the stage you want to start from in the experiments. This allows you to skip certain stages, such as data preparation, if they have already been completed.

`stop_stage` denotes the number of the stage you want to end at in the experiments. This is useful when you only want to run a subset of the stages.

`conf_path` denotes the path to the configuration file of the model. This file contains all the necessary parameters and settings for the model.

`avg_num` denotes the number K of top-K models you want to average to get the final model. Averaging multiple models can improve the robustness and performance of the final model.

`decode_conf_path` denotes the path to the configuration file used for decoding. This file contains settings related to decoding, such as beam size and language model weight.

`audio_file` denotes the file path of the single audio file you want to infer in stage 5. This is useful for testing the model on a specific audio file.

`ckpt` denotes the checkpoint prefix of the model. This is the name of the directory under which the model checkpoints are saved, e.g., "conformer". Note that you cannot set this variable via the command line; it is derived from the `conf_path`.

`ips` (optional) can be used to specify the IP addresses of multiple machines for distributed training. This is not typically used in single-machine setups.

You can set the local variables (except `ckpt`) when you use the `run.sh` script via command-line options. For example, you can set the `gpus` and `avg_num` when you use the following command:

```bash
bash run.sh --gpus 0,1 --avg_num 20
```

The script uses the `parse_options.sh` utility to parse these command-line options and set the corresponding variables. The script then proceeds to execute the stages specified by `stage` and `stop_stage`, using the configured variables throughout the process.
## Stage 0: Data Processing
To use this example, you need to process the data first. Stage 0 in the `run.sh` script is dedicated to this task. The relevant code snippet is shown below:

```bash
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi
```

Stage 0 is specifically for processing the data. This stage prepares all necessary datasets and metadata required for subsequent training and evaluation phases.

If you only want to process the data without proceeding to other stages, you can run the following command:

```bash
bash run.sh --stage 0 --stop_stage 0
```

Alternatively, you can manually execute the data processing script in your command line. Ensure you have sourced the necessary environment setup scripts first:

```bash
source path.sh
bash ./local/data.sh
```

After successfully processing the data, the `data` directory will be populated with the following structure:

```bash
data/
|-- dev.meta
|-- lang_char
|   `-- bpe_unigram_5000.model
|   `-- bpe_unigram_5000.vocab
|   `-- vocab.txt
|-- manifest.dev
|-- manifest.dev.raw
|-- manifest.test
|-- manifest.test.raw
|-- manifest.train
|-- manifest.train.raw
|-- mean_std.json
|-- test.meta
`-- train.meta
```

This directory structure contains manifests, metadata files, vocabulary files, and mean-std normalization parameters necessary for the training and evaluation of your model. Each file and directory serves a specific purpose and is essential for the pipeline to function correctly.
## Stage 1: Model Training
If you want to train the model, you can use stage 1 in the `run.sh` script. The relevant code segment is shown below:
```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt} ${ips}
fi
```

To train the model, you can use the following command to execute stages 0 and 1:
```bash
bash run.sh --stage 0 --stop_stage 1
```

Alternatively, you can run the necessary scripts directly in the command line. If you only want to use the CPU, you can use:
```bash
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer.yaml transformer
```

If you want to use GPUs (assuming you have multiple GPUs configured in the `gpus` variable, e.g., `gpus=0,1,2,3`), you can specify the GPU devices as follows:
```bash
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ./local/train.sh conf/transformer.yaml transformer <ips>
```

Remember to replace `<ips>` with the actual IP addresses of the machines if you are running the training in a distributed setting. If you are running on a single machine, you can leave the `ips` variable empty in the `run.sh` script.

By running the above commands, the script will prepare the data (if `stage 0` is included), and then proceed to train the model (stage 1). The trained checkpoints will be saved under the `exp` directory.
## Stage 2: Top-k Models Averaging

After training the model, we need to get the final model for testing and inference. In every epoch, the model checkpoint is saved, so we can choose the best model from them based on the validation loss or we can sort them and average the parameters of the top-k models to get the final model. Averaging the top-k models often leads to a more robust and generalizable final model.

We can use stage 2 to perform this model averaging, and the relevant code snippet is shown below:

```bash
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi
```

Here, `avg.sh` is a script located in the `../../../utils/` directory, which is defined in the `path.sh` script. This script is responsible for averaging the parameters of the top-k models. The `${ckpt}` variable represents the basename of the configuration file (excluding the extension), and `${avg_num}` specifies the number of top models to average.

To execute this stage along with the previous stages (stage 0 for data preparation and stage 1 for model training), you can use the following command:

```bash
bash run.sh --stage 0 --stop_stage 2
```

Alternatively, you can run the individual scripts in the command line (CPU-only) using the following sequence:

```bash
source path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer.yaml transformer
avg.sh best exp/transformer/checkpoints 30
```

In this example, `conf/transformer.yaml` is the configuration file for the transformer model, `transformer` is the basename used for the checkpoints, and `30` is the number of top models to average.

Make sure to adjust the `conf_path`, `ckpt`, `gpus`, and `avg_num` variables in the main script according to your specific setup and requirements. The `avg_ckpt` variable will be used in subsequent stages for testing and inference with the averaged model.
## Stage 3: Model Testing
The test stage is designed to evaluate the model performance. This stage uses the averaged checkpoint (avg_n) obtained from the previous stage to assess the model's accuracy and reliability. The code snippet responsible for this stage is provided below:
```bash
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # test ckpt avg_n
    CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
```

Here's a breakdown of the script:
- `CUDA_VISIBLE_DEVICES=0` specifies that only the first GPU (with index 0) should be used for testing.
- `./local/test.sh` is the script that performs the testing.
- `${conf_path}` is the path to the configuration file that defines the model architecture and other training parameters.
- `${decode_conf_path}` is the path to the decoding configuration file that includes parameters like beam size and language model weight.
- `exp/${ckpt}/checkpoints/${avg_ckpt}` specifies the location of the averaged checkpoint to be tested.

If you want to train a model from scratch and test it up to stage 3, you can use the following script:
```bash
bash run.sh --stage 0 --stop_stage 3
```

Alternatively, you can manually run the relevant scripts in the command line. Below is an example using only the CPU (by omitting the `CUDA_VISIBLE_DEVICES` setting):
```bash
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
# Assuming you have set the `gpus`, `conf_path`, `decode_conf_path`, `avg_num`, and other variables correctly
# Train the model
# (Note: This step is omitted here for brevity, but it involves running `./local/train.sh` with appropriate arguments)
# Average the best models
avg.sh best exp/${ckpt}/checkpoints ${avg_num}
# Test the averaged checkpoint
CUDA_VISIBLE_DEVICES= ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt}
```

Remember to customize the paths and parameters according to your specific setup and model configuration.
## Pretrained Model
You can get the pretrained transformer or conformer models from [this](../../../docs/source/released_model.md) link.

After downloading the pretrained model, you can use the `tar` command to unpack it. Once unpacked, you can leverage a series of scripts to handle various tasks such as training, testing, aligning, and exporting the model.

Here's a detailed guide on how to use these scripts:

### Unpacking the Pretrained Model
```bash
wget https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
tar xzvf asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
source path.sh
```

### Data Preparation
If you haven't processed the data and obtained the manifest file, you need to run the following commands to prepare the data:
```bash
bash local/data.sh --stage -1 --stop_stage -1
bash local/data.sh --stage 2 --stop_stage 2
```
These steps are optional if you already have the necessary data prepared.

### Training the Model
You can train the model using the provided training script. Set the GPU devices, configuration file path, and other parameters as needed:
```bash
gpus=0,1,2,3
conf_path=conf/transformer.yaml
stage=0
stop_stage=1

CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt_name} ${ips}
```
Replace `${ckpt_name}` with an appropriate name for your checkpoint and `${ips}` with the IP addresses of the nodes (if you're using multiple nodes for distributed training).

### Averaging Best Models
After training, you can average the best models to improve performance:
```bash
avg_num=30
stage=2
stop_stage=2

avg.sh best exp/${ckpt_name}/checkpoints ${avg_num}
```

### Testing the Model
You can test the averaged model using the testing script:
```bash
decode_conf_path=conf/tuning/decode.yaml
avg_ckpt=avg_${avg_num}
stage=3
stop_stage=3

CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt_name}/checkpoints/${avg_ckpt}
```

### CTC Alignment
To perform CTC alignment on the test data:
```bash
stage=4
stop_stage=4

CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} exp/${ckpt_name}/checkpoints/${avg_ckpt}
```

### Testing a Single WAV File
You can also test a single WAV file using the provided script:
```bash
audio_file=data/demo_002_en.wav
stage=5
stop_stage=5

CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt_name}/checkpoints/${avg_ckpt} ${audio_file}
```

### Exporting the Model
Finally, you can export the model for inference:
```bash
stage=51
stop_stage=51

CUDA_VISIBLE_DEVICES= ./local/export.sh ${conf_path} exp/${ckpt_name}/checkpoints/${avg_ckpt} exp/${ckpt_name}/checkpoints/${avg_ckpt}.jit
```

The performance of the released models are shown in [this](./RESULTS.md) document.
## Stage 4: CTC Alignment

This stage is dedicated to obtaining the alignment between the audio and the text using Connectionist Temporal Classification (CTC). CTC is a powerful technique that aligns the input sequence to the target sequence without needing an alignment pre-specified.

```bash
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # ctc alignment of test data
    CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
```

To perform CTC alignment, you need to have a trained model and its average checkpoint. The script above will use the specified configuration file (`conf_path`), decoding configuration (`decode_conf_path`), and the average checkpoint (`avg_ckpt`) located in the `exp/${ckpt}/checkpoints/` directory.

If you have already trained a model and wish to obtain the alignment for your test data, you can use the following commands. These commands assume you have already set up your environment and paths correctly:

```bash
# Assuming you have already trained the model and obtained the average checkpoint
bash run.sh --stage 0 --stop_stage 4
```

Alternatively, if you only need to perform the alignment without retraining the model, you can specify the stages directly:

```bash
# Load necessary environment variables
. ./path.sh
. ./cmd.sh

# Assuming `conf_path` and `decode_conf_path` are already set correctly
CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt}
```

In the script, `CUDA_VISIBLE_DEVICES=0` specifies that the alignment process should run on the first GPU. If you do not have a GPU or want to use a different one, you can adjust this setting accordingly.

Note that the `align.sh` script will generate alignment information for your test data, which can be useful for various applications such as visualization, error analysis, and more.

If you encounter any issues during this stage, ensure that all dependencies are correctly installed, and that the paths to the configuration files and checkpoints are correct. Additionally, check the logs for any error messages that might provide insight into the problem.
## Stage 5: Single Audio File Inference

In some situations, you may want to use the trained model to perform inference on a single audio file. This can be accomplished using Stage 5. The relevant code snippet is shown below:

```bash
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # test a single .wav file
    CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file} || exit -1
fi
```

Before running this script, you have the option to train the model yourself using the following command:

```bash
bash run.sh --stage 0 --stop_stage 3
```

Alternatively, you can download a pretrained model using the script below. Please note that the configuration file and model checkpoint may vary depending on the specific experiment you are running. For this example, let's assume you are using a Conformer model trained on Librispeech:

```bash
wget https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
tar xzvf asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
```

You can also download a sample audio file to test the inference:

```bash
wget -nc https://paddlespeech.bj.bcebos.com/datasets/single_wav/en/demo_002_en.wav -P data/
```

Please ensure that your audio file, whether it's the sample provided or your own, has a sample rate of 16K. To run the inference on the sample audio file, you can use the following command:

```bash
CUDA_VISIBLE_DEVICES= ./local/test_wav.sh conf/conformer.yaml conf/tuning/decode.yaml exp/conformer/checkpoints/avg_20 data/demo_002_en.wav
```

In this command:
- `conf/conformer.yaml` is the configuration file for the Conformer model.
- `conf/tuning/decode.yaml` contains decoding parameters.
- `exp/conformer/checkpoints/avg_20` is the path to the averaged checkpoint of the trained model.
- `data/demo_002_en.wav` is the path to the audio file you want to perform inference on.

Make sure to adjust the paths and filenames according to your specific setup. If the inference runs successfully, you should see the transcription result of the audio file printed in the console or saved to a file (depending on how the `test_wav.sh` script is implemented).
## Stage 5: Single Audio File Testing
The single audio file testing stage is designed to evaluate the model's performance on a specific audio file. The code for this stage is shown below:
```bash
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # test a single .wav file
    CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file} || exit -1
fi
```
To perform the single audio file testing, you can use the script below to execute stages up to and including stage 5:
```bash
bash run.sh --stage 0 --stop_stage 5
```
Alternatively, you can manually run these scripts in the command line (using only CPU for inference if desired). Below is an example of the full sequence of commands:
```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh ${conf_path} ${ckpt} ${ips}
avg.sh best exp/${ckpt}/checkpoints ${avg_num}
CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt}
CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt}
CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file}
```

Remember to replace `${conf_path}`, `${ckpt}`, `${ips}`, `${decode_conf_path}`, `${avg_num}`, and `${audio_file}` with your actual configuration file path, checkpoint name, IP addresses, decode configuration file path, average number of models, and the path to your audio file, respectively.
