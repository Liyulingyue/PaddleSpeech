# Transformer/Conformer ASR with Librispeech
This example contains code used to train [u2](https://arxiv.org/pdf/2012.05481.pdf) model (Transformer or [Conformer](https://arxiv.org/pdf/2005.08100.pdf) model) with [Librispeech dataset](http://www.openslr.org/resources/12)

## Overview
All the scripts you need are in the `run.sh`. There are several stages in the `run.sh`, and each stage has its function.

| Stage | Function                                                     |
|:---- |:----------------------------------------------------------- |
| 0     | Process data. It includes: <br>       (1) Download the dataset <br>       (2) Calculate the CMVN of the train dataset <br>       (3) Get the vocabulary file <br>       (4) Get the manifest files of the train, development, and test datasets <br>       (5) Get the sentencepiece model |
| 1     | Train the model                                              |
| 2     | Get the final model by averaging the top-k models, setting k = 1 means choosing the best model                               |
| 3     | Test the final model performance                             |
| 4     | Get CTC alignment of test data using the final model          |
| 5     | Infer a single audio file                                    |
| 51    | Export the final model as a JIT (Just-In-Time) compiled static graph |

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
The document below will describe the scripts in the `run.sh` in detail.
## The Environment Variables
The `path.sh` contains the essential environment variables required for the system to function correctly. 
```bash
. ./path.sh
. ./cmd.sh
```
These scripts need to be sourced first to ensure all necessary paths and variables are set up properly. 

Additionally, another script is also required:
```bash
source ${MAIN_ROOT}/utils/parse_options.sh
```
This script enhances the shell scripts by enabling the use of `--variable value` options.

The environment variables set in `path.sh` and `cmd.sh` typically include paths to directories, executable files, and other configuration settings. These variables are crucial for scripts like data preparation, model training, evaluation, and export.

Here are some key environment variables that might be set in `path.sh` and used throughout the scripts:

- `MAIN_ROOT`: The root directory of the project.
- `gpus`: A comma-separated list of GPU IDs to use for training and evaluation.
- `stage` and `stop_stage`: Control the stages of the process to run, enabling partial execution for debugging or testing purposes.
- `conf_path`: The path to the configuration file for the model.
- `decode_conf_path`: The path to the decoding configuration file.
- `avg_num`: The number of best models to average for better performance.
- `audio_file`: A path to an audio file for testing the model with a single input.

Here is an example of how these variables are used in the script:
```bash
#!/bin/bash
set -e

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

gpus=0,1,2,3
stage=0
stop_stage=50
conf_path=conf/transformer.yaml
ips=            #xx.xx.xx.xx,xx.xx.xx.xx (fill with actual IP addresses)
decode_conf_path=conf/tuning/decode.yaml
avg_num=30
audio_file=data/demo_002_en.wav

. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

avg_ckpt=avg_${avg_num}
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
echo "checkpoint name ${ckpt}"

# Prepare data, train model, average best models, test, align, test a single .wav file, and export model
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    bash ./local/data.sh || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt} ${ips}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ./local/avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file} || exit -1
fi

if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ]; then
    CUDA_VISIBLE_DEVICES= ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit
fi
```

Ensure that `path.sh` and `cmd.sh` are sourced correctly at the beginning of your scripts to avoid errors related to missing environment variables.
## The Local Variables

Some local variables are set in the `run.sh` script and within the bash script itself for configuring the experiment. Here's a detailed breakdown of each variable:

`gpus` denotes the GPU number(s) you want to use. If you set `gpus=` (an empty value), it means you only use the CPU. Multiple GPUs can be specified with a comma-separated list, e.g., `0,1,2,3`.

`stage` denotes the number of the stage you want to start from in the experiments. This allows you to skip certain stages if they have already been completed.

`stop_stage` denotes the number of the stage you want to end at in the experiments. This is useful for running only a subset of the stages.

`conf_path` denotes the config path of the model. This should point to a YAML file containing the model configuration.

`avg_num` denotes the number K of top-K models you want to average to get the final model. Averaging multiple models can improve performance and robustness.

`decode_conf_path` denotes the path to the decoding configuration file, which is used during the testing phase.

`ips` (not typically set via `run.sh`) allows you to specify IP addresses for distributed training. This variable is commented out by default in the script.

`audio_file` denotes the file path of the single audio file you want to infer in stage 5. This is useful for evaluating the model on a specific audio sample.

`ckpt` denotes the checkpoint prefix of the model, e.g., "conformer". This prefix is used to identify the model checkpoints during training and evaluation.

You can set the local variables (except `ckpt`, which is derived from `conf_path`) when you use the `run.sh` script. The script uses `parse_options.sh` to handle command-line arguments and update the variables accordingly.

For example, you can set the `gpus` and `avg_num` when you use the command line:
```bash
bash run.sh --gpus 0,1 --avg_num 20
```

The bash script also includes logic to handle different stages of the experiment, such as preparing data, training the model, averaging the best models, testing the averaged model, aligning test data, testing a single `.wav` file, and exporting the model for inference. Each stage is conditionally executed based on the `stage` and `stop_stage` variables.
## Stage 0: Data Processing and Model Training

If you want to process the data and train the model, you can use stages 0 and 1 in the `run.sh`. The code is shown below:

```bash
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt} ${ips}
fi
```

To process the data and train the model, you can use the script below to execute stages 0 and 1:

```bash
bash run.sh --stage 0 --stop_stage 1
```

Alternatively, you can run these scripts in the command line step-by-step. First, source the configuration scripts and process the data:

```bash
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
```

After processing the data, the `data` directory will look like this:

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

To train the model, set the `CUDA_VISIBLE_DEVICES` environment variable to specify the GPUs you want to use (e.g., `0,1,2,3` for multiple GPUs or `0` for a single GPU), and run the training script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./local/train.sh ${conf_path} ${ckpt} ${ips}
```

If you only have one GPU or want to use CPU, adjust the `CUDA_VISIBLE_DEVICES` accordingly:

```bash
CUDA_VISIBLE_DEVICES=0 ./local/train.sh ${conf_path} ${ckpt} ${ips}  # For a single GPU
# or
CUDA_VISIBLE_DEVICES= ./local/train.sh ${conf_path} ${ckpt} ${ips}  # For CPU (though it's much slower)
```

Replace `${conf_path}`, `${ckpt}`, and `${ips}` with your actual configuration file path, checkpoint name, and IP addresses for distributed training, respectively.
## Stage 1: Top-k Models Averaging

After training the model in Stage 1, we proceed to average the parameters of the top-k best models to improve robustness and generalization. This technique, often referred to as model averaging or ensemble averaging, combines multiple models to reduce variance and improve overall performance.

In our pipeline, each epoch saves a model checkpoint. We can sort these checkpoints based on validation loss and select the top-k models for averaging. This process is automated in Stage 1, and the relevant code is shown below:

```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt} ${ips}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi
```

Here, the `avg.sh` script is used to average the top-k models. This script is located in the `../../../utils/` directory, which is sourced in the `path.sh` file. The `avg_num` variable specifies the number of top models to average.

To execute Stage 1 and proceed with model averaging, you can use the following command:

```bash
bash run.sh --stage 1 --stop_stage 2
```

Alternatively, you can run the scripts in the command line manually (using only CPU if desired):

```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh ${conf_path} ${ckpt}
avg.sh best exp/${ckpt}/checkpoints ${avg_num}
```

This will train the model, save the checkpoints, and then average the top-k models specified by `avg_num`. The averaged model will be used in subsequent stages for testing and inference.
## Stage 2: Model Testing

After averaging the top-k models, we proceed to the testing stage to evaluate the performance of the final averaged model. The testing stage ensures that the model works well on unseen data and provides insights into its accuracy and robustness. The code for the testing stage is shown below:

```bash
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # test ckpt avg_n
    CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
```

In this script, the `test.sh` script is responsible for evaluating the model. It reads the configuration file specified by `${conf_path}`, decoding configuration file `${decode_conf_path}`, and the checkpoint directory of the averaged model `exp/${ckpt}/checkpoints/${avg_ckpt}`. The script runs the evaluation on the GPU specified by `CUDA_VISIBLE_DEVICES=0`.

If you want to train a model, average the top-k models, and test the final averaged model, you can use the script below to execute stage 0, stage 1, stage 2, and stage 3:

```bash
bash run.sh --stage 0 --stop_stage 3
```

Alternatively, you can run these scripts step-by-step in the command line (only using CPU if needed):

```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer.yaml transformer xx.xx.xx.xx,xx.xx.xx.xx
avg.sh best exp/transformer/checkpoints 30
CUDA_VISIBLE_DEVICES=0 ./local/test.sh conf/transformer.yaml conf/tuning/decode.yaml exp/transformer/checkpoints/avg_30
```

This will prepare the data, train the model, average the top-k models, and finally test the averaged model using the `test.sh` script. The output of the test stage will provide you with detailed performance metrics, such as word error rate (WER) or character error rate (CER), depending on your evaluation setup.
## Pretrained Model
You can get the pretrained transformer or conformer models from [this](../../../docs/source/released_model.md) page.

Once you have downloaded the model, you can use the `tar` command to unpack it. After unpacking, you can leverage a series of bash scripts to train, average the best models, test the model, align the test data, test a single `.wav` file, and even export the model for inference.

Here's a step-by-step guide to utilizing these scripts:

1. **Download and Unpack the Model**:

    ```bash
    wget https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
    tar xzvf asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
    source path.sh
    ```

2. **Prepare Data (if not already processed)**:

    If you haven't processed your data and generated the manifest file, you need to run the following commands to prepare the data:

    ```bash
    bash local/data.sh --stage -1 --stop_stage -1
    bash local/data.sh --stage 2 --stop_stage 2
    ```

3. **Training the Model**:

    You can train the model using multiple GPUs by specifying the `gpus` variable. Adjust the `conf_path` to the configuration file you want to use.

    ```bash
    gpus=0,1,2,3
    stage=0
    stop_stage=1
    conf_path=conf/transformer.yaml
    
    . ${MAIN_ROOT}/utils/parse_options.sh
    
    ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
    
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt}
    ```

4. **Average the Best Models**:

    After training, you can average the best `n` models to improve performance.

    ```bash
    stage=2
    stop_stage=2
    avg_num=30
    
    . ${MAIN_ROOT}/utils/parse_options.sh
    
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
    ```

5. **Test the Averaged Model**:

    Use the test script to evaluate the averaged model.

    ```bash
    stage=3
    stop_stage=3
    decode_conf_path=conf/tuning/decode.yaml
    avg_ckpt=avg_${avg_num}
    
    . ${MAIN_ROOT}/utils/parse_options.sh
    
    CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt}
    ```

6. **CTC Alignment of Test Data**:

    If you need to align the test data using CTC, you can run:

    ```bash
    stage=4
    stop_stage=4
    
    . ${MAIN_ROOT}/utils/parse_options.sh
    
    CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt}
    ```

7. **Test a Single `.wav` File**:

    To test a single audio file, use the `test_wav.sh` script.

    ```bash
    stage=5
    stop_stage=5
    audio_file=data/demo_002_en.wav
    
    . ${MAIN_ROOT}/utils/parse_options.sh
    
    CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file}
    ```

8. **Export the Model**:

    Finally, you can export the model for inference.

    ```bash
    stage=51
    stop_stage=51
    
    . ${MAIN_ROOT}/utils/parse_options.sh
    
    CUDA_VISIBLE_DEVICES= ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit
    ```

The performance of the released models are shown in [this](./RESULTS.md).
## Stage 3: CTC Alignment

The CTC Alignment stage aims to generate the alignment between the audio and the text. This step is crucial for understanding how the model maps audio signals to text characters. The code for this stage is provided below:

```bash
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # ctc alignment of test data
    CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
```

To perform CTC alignment, you need to have already trained and averaged your model. The script above uses the averaged checkpoint (`avg_ckpt`) to generate the alignment.

If you want to train the model, test it, and perform the alignment in sequence, you can use the following script to execute stages 0 through 4:

```bash
bash run.sh --stage 0 --stop_stage 4
```

Alternatively, if you have already trained and averaged your model but skipped the test stage, you can directly proceed to the alignment stage using:

```bash
bash run.sh --stage 4 --stop_stage 4
```

Or, you can manually run the necessary scripts in the command line. Below is an example using only the CPU:

```bash
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
# Assuming you have already trained and averaged your model
# CUDA_VISIBLE_DEVICES= ./local/train.sh and avg.sh commands would be run previously
# Now, proceed to the alignment stage
CUDA_VISIBLE_DEVICES= ./local/align.sh conf/transformer.yaml conf/tuning/decode.yaml exp/${ckpt}/checkpoints/${avg_ckpt}
```

Make sure to replace `conf/transformer.yaml` and other paths with the actual configuration and checkpoint paths you are using. The `decode_conf_path` is also crucial as it contains parameters related to decoding, which might affect the alignment result.
## Stage 4: Export Model to JIT Format
This stage involves exporting the trained model to a JIT (Just-In-Time) format, which is optimized for inference. The JIT format allows for faster and more efficient model loading and execution.

```bash
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # export ckpt avg_n to JIT format
    CUDA_VISIBLE_DEVICES= ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit
fi
```

If you have successfully trained and averaged your model in the previous stages, you can now export it to the JIT format by running the script above. The script takes the configuration file path, the path to the averaged checkpoint, and the desired output path for the JIT model as arguments.

Here is an example command to export a model:

```bash
source path.sh
./local/export.sh conf/transformer.yaml exp/transformer/checkpoints/avg_30 exp/transformer/checkpoints/avg_30.jit
```

In this example, `conf/transformer.yaml` is the configuration file used for training, `exp/transformer/checkpoints/avg_30` is the path to the averaged checkpoint, and `exp/transformer/checkpoints/avg_30.jit` is the desired output path for the JIT model.

Make sure to adjust the paths and filenames according to your specific setup. Once the export process is complete, you will have a JIT model ready for inference.
## Stage 5: Single Audio File Inference

In some situations, you want to use the trained model to perform inference on a single audio file. You can use stage 5 for this purpose. The code segment related to this stage is shown below:

```bash
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Test a single .wav file
    CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file} || exit -1
fi
```

To use this script, you have the option to train the model yourself using the following command:

```bash
bash run.sh --stage 0 --stop_stage 3
```

Alternatively, you can download a pretrained model using the script below. Make sure to adjust the URL to match the specific model you want to use:

```bash
wget https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
tar xzvf asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
```

You can also download a demo audio file to test the inference:

```bash
wget -nc https://paddlespeech.bj.bcebos.com/datasets/single_wav/en/demo_002_en.wav -P data/
```

Please ensure that your audio file, whether the demo or your own, has a sample rate of 16K. To run the inference on the demo audio file, use the following command:

```bash
CUDA_VISIBLE_DEVICES= ./local/test_wav.sh conf/conformer.yaml conf/tuning/decode.yaml exp/conformer/checkpoints/avg_20 data/demo_002_en.wav
```

In this command:
- `conf/conformer.yaml` is the configuration file for the model.
- `conf/tuning/decode.yaml` contains decoding-related configurations.
- `exp/conformer/checkpoints/avg_20` is the path to the averaged checkpoint of the trained model.
- `data/demo_002_en.wav` is the path to the audio file you want to test.

Adjust these paths according to your specific setup and the model you are using.
