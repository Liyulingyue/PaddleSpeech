# 使用 Aishell 数据集训练 DeepSpeech2 的离线/在线 ASR 模型
此示例包含用于使用[Aishell数据集](http://www.openslr.org/resources/33)训练DeepSpeech2流式或非流式模型的代码。
## 概述
您需要的所有脚本都在`run.sh`中。`run.sh`中有几个阶段，每个阶段都有其功能。
| 阶段 | 功能 |
|:---- |:----------------------------------------------------------- |
| 0 | 数据处理。包括：<br> (1) 下载数据集 <br> (2) 计算训练数据集的CMVN <br> (3) 获取词汇文件 <br> (4) 获取训练、开发和测试数据集的manifest文件 |
| 1 | 训练模型 |
| 2 | 通过平均最好k个最佳模型来获得最终模型，设置k=1表示选择最佳模型 |
| 3 | 测试最终模型性能 |
| 4 | 导出静态图模型 |
| 5 | 测试静态图模型 |
| 6 | 对单个音频文件进行推理 |
您可以通过设置`stage`和`stop_stage`来选择运行一系列阶段。
例如，如果您想执行阶段2和阶段3的代码，可以运行以下脚本：
```bash
bash run.sh --stage 2 --stop_stage 3
```
或者，您可以将`stage`设置为等于`stop-stage`来仅运行一个阶段。
例如，如果您只想运行`stage 0`，可以使用以下脚本：
```bash
bash run.sh --stage 0 --stop_stage 0
```
下面的文档将详细描述`run.sh`中的脚本。
## 环境变量
`path.sh`包含环境变量。
```bash
source path.sh
运行程序前需要先执行此脚本。
另一个同样需要执行的脚本：
```bash
source ${MAIN_ROOT}/utils/parse_options.sh
```
它将支持在shell脚本中使用`--variable value`的方式。
## 局部变量
在`run.sh`中设置了一些局部变量。
`gpus`表示您想使用的GPU数量。如果您设置`gpus=`，则表示仅使用CPU。
`stage`表示您想在实验中从哪个阶段开始。
`stop_stage`表示您想在实验中结束于哪个阶段。
`conf_path`表示模型的配置路径。
`avg_num`表示要平均的最好的 k 个最佳模型的数量，以获得最终模型。
`model_type`表示模型类型：流式/非流式
`audio_file`表示在阶段6中您想进行推理的单个文件的路径。
`ckpt`表示模型的检查点前缀，例如"deepspeech2"
您可以在使用`run.sh`时设置局部变量（除了`ckpt`）。
例如，您可以在命令行中设置`gpus`和`avg_num`：
```bash
bash run.sh --gpus 0,1 --avg_num 1
```
## 阶段0：数据处理
要使用此示例，您需要先处理数据，可以使用`run.sh`中的阶段0来完成此操作。代码如下：
```bash
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
 # 准备数据
 bash ./local/data.sh || exit -1
fi
```
阶段0用于处理数据。
如果您只想处理数据，可以运行：
```bash
bash run.sh --stage 0 --stop_stage 0
```
您也可以直接在命令行中运行这些脚本。
```bash
source path.sh
bash ./local/data.sh
```
处理完数据后，`data`目录将如下所示：
```bash
data/
|-- dev.meta
|-- lang_char
| `-- vocab.txt
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
## 阶段1：模型训练
如果您想训练模型，可以使用`run.sh`中的阶段1。代码如下：
```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
 # 训练模型，所有`ckpt`在`exp`目录下
 CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt}
fi
```
如果您想训练模型，可以使用以下脚本来执行阶段0和阶段1：
```bash
bash run.sh --stage 0 --stop_stage 1
```
或者，您可以在命令行中运行这些脚本（仅使用CPU）。
```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/deepspeech2.yaml deepspeech2
```
如果您想使用GPU，可以在命令行中运行这些脚本（假设您只有一个GPU）。
```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES=0 ./local/train.sh conf/deepspeech2.yaml deepspeech2
```
## 阶段2：前k个模型平均
训练完模型后，我们需要得到用于测试和推理的最终模型。在每个epoch中，都会保存模型检查点，因此我们可以基于验证损失从它们中选择最佳模型，或者我们可以对它们进行排序并平均前k个模型的参数以得到最终模型。我们可以使用阶段2来完成此操作，代码如下：
```bash
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
 # 平均n个最佳模型
 avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi
```
`avg.sh`在`../../../utils/`中定义，该路径在`path.sh`中定义。
如果您想得到最终模型，可以使用以下脚本来执行阶段0、阶段1和阶段2：
```bash
bash run.sh --stage 0 --stop_stage 2
```
或者，您可以在命令行中运行这些脚本（仅使用CPU）。
```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/deepspeech2.yaml deepspeech2
avg.sh best exp/deepspeech2/checkpoints 1
```
## 阶段3：模型测试
测试阶段用于评估模型性能。测试阶段的代码如下：
```bash
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
 # 测试ckpt avg_n
 CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
```
如果您想训练并测试模型，可以使用以下脚本来执行阶段0、阶段1、阶段2和阶段3：
```bash
bash run.sh --stage 0 --stop_stage 3
```
或者，您可以在命令行中运行这些脚本（仅使用CPU）。
```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/deepspeech2.yaml deepspeech2
avg.sh best exp/deepspeech2/checkpoints 1
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/deepspeech2.yaml conf/tuning/decode.yaml exp/deepspeech2/checkpoints/avg_10
```
## 预训练模型
您可以从[这里](../../../docs/source/released_model.md)获取预训练模型。
使用`tar`脚本来解压模型，然后您可以使用脚本来测试模型。
例如：
```bash
wget https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_offline_aishell_ckpt_1.0.1.model.tar.gz
tar xzvf asr0_deepspeech2_offline_aishell_ckpt_1.0.1.model.tar.gz
source path.sh
# 如果您已经处理了数据并获得了manifest文件，则可以跳过以下两个步骤
bash local/data.sh --stage -1 --stop_stage -1
bash local/data.sh --stage 2 --stop_stage 2
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/deepspeech2.yaml exp/deepspeech2/checkpoints/avg_10
```
发布的模型性能如[这里](./RESULTS.md)所示。
## 阶段4：导出静态图模型
此阶段是将动态图转换为静态图。
```bash
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
 # 导出ckpt avg_n
 CUDA_VISIBLE_DEVICES=0 ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit ${model_type}
fi
```
如果您已经有一个动态图模型，可以运行以下脚本：
```bash
source path.sh
./local/export.sh conf/deepspeech2.yaml exp/deepspeech2/checkpoints/avg_10 exp/deepspeech2/checkpoints/avg_10.jit
```
## 阶段5：测试静态图模型
与阶段3类似，静态图模型也可以进行测试。
```bash
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
 # 测试导出的ckpt avg_n
 CUDA_VISIBLE_DEVICES=0 ./local/test_export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt}.jit ${model_type}|| exit -1
fi
```
如果您已经导出了静态图，可以运行以下脚本：
```bash
CUDA_VISIBLE_DEVICES= ./local/test_export.sh conf/deepspeech2.yaml conf/tuning/decode.yaml exp/deepspeech2/checkpoints/avg_10.jit
```
## 阶段6：单个音频文件推理
在某些情况下，您可能想使用训练好的模型对单个音频文件进行推理。您可以使用阶段5。代码如下：
```bash
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
 # 测试单个.wav文件
 CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${model_type} ${audio_file}
fi
```
您可以自己训练模型，或者下载预训练模型，使用以下脚本：
```bash
wget https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_offline_aishell_ckpt_1.0.1.model.tar.gz
tar asr0_deepspeech2_offline_aishell_ckpt_1.0.1.model.tar.gz
```
您可以下载音频演示文件：
```bash
wget -nc https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/demo_01_03.wav -P data/
```
您需要准备一个音频文件或使用上述音频演示文件，请确认音频的采样率为16K。您可以通过运行以下脚本来获取音频演示文件的结果。
```bash
CUDA_VISIBLE_DEVICES= ./local/test_wav.sh conf/deepspeech2.yaml conf/tuning/decode.yaml exp/deepspeech2/checkpoints/avg_10 data/demo_01_03.wav
```
