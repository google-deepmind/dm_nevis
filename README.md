<div align="center">

# 🏝️ NEVIS'22

[![Paper](https://img.shields.io/badge/arXiv-2211.11747-red)](https://arxiv.org/abs/2211.11747)
[![Blog](https://img.shields.io/badge/blog-link-blue)](https://www.deepmind.com/blog/benchmarking-the-next-generation-of-never-ending-learners)

</div>

NEVIS’22 is a benchmark for measuring the performance of algorithms in the field
of continual learning. Please see the accompanying [paper] for more details.

Within this Python package, we provide three components,

1.  Library code to download and post-process datasets that are not available
    within [tfds], so that the stream used in the [paper] can be replicated.
2.  A package to combine the NEVIS’22 datasets into a *stream*, and robustly
    evaluate learners using the evaluation protocol proposed in the NEVIS’22
    [paper].
3.  Baseline learners implemented in JAX and PyTorch. The JAX learners are
    identical to the learners used for the figures in the [paper], the PyTorch
    learners are provided for example purposes.

NEVIS’22 is composed of 106 tasks chronologically sorted and extracted from
publications randomly sampled from online proceedings of major computer vision
conferences over the past three decades. Each task is a supervised
classification task, which is the most well understood setting in machine
learning. The challenge is how to automatically transfer knowledge across
related tasks in order to achieve a higher performance or be more efficient on
the next task.

By construction, NEVIS’22 is reproducible, diverse and at a scale sufficiently
large to test state of the art learning algorithms. The task selection process
does not favor any particular approach, but merely tracks what the computer
vision community has deemed interesting over time. NEVIS’22 is not just about
data, it is also about the methodology used to train and evaluate learners. We
evaluate learners in terms of their ability to learn future tasks, as measured
by their trade-off between error rate and compute measured in the number of
floating-point operations. In NEVIS’22, achieving lower error rate is by itself
not sufficient, if this comes at an unreasonable computational cost. Instead, we
incentivise both accurate and efficient models.

You can read more about NEVIS'22 in our [paper] and our [blog post].

## 0. Dependencies

Please follow these steps and read in details section 1. and 2. before launching
anything.

-   Our datasets use the Tensorflow(-datasets) API. Our JAX learners use
    TensorFlow and JAX, and our PyTorch Learners use PyTorch. Each (datasets,
    jax learners, and pytorch learners) have their own `requirements.txt` that
    you are welcome to install with `pip` and a Python version above 3.8.

-   It is also possible to run the code directly using the provided Dockerfiles.
    see [here](https://docs.docker.com/get-docker/) for installing Docker.

-   Some datasets are downloaded from Kaggle. See the
    [Kaggle website](https://www.kaggle.com/docs/api) for configuring your
    credentials, and place them in the folder ~/.kaggle.

## 1. Replicating the NEVIS'22 stream

In NEVIS'22, we train and evaluate on *streams*. Each stream is a sequence of
datasets. Some streams have a large number of datasets, up to 106, allowing us
to evaluate Large-Scale Continual Learning.

There are three different sources for datasets in NEVIS'22:

1.  **Datasets on Tensorflow-Datasets (TFDS)**: they will be downloaded
    automatically when needed

2.  **Custom dataset downloaders**: you need the `./build_dataset.sh` script

3.  **Manual dataset download**: you need to download data yourself

Note that we do not host or distribute these datasets, instead we provide URLS
to their original source to help you download them at your own risk. We do not
vouch for their quality or fairness, or claim that you have license to use the
dataset. It is your responsibility to determine whether you have permission to
use the dataset under the dataset's license. If you're a dataset owner and wish
to update any part of it (description, citation, etc.), or do not want your
dataset URL to be included in this library, please get in touch through a GitHub
issue. Thanks for your contribution to the ML community!

We do our best to keep datasets URLs up-to-date. If a dataset doesn't download,
please contact the dataset owners and open an issue to let us know. If a dataset
doesn't get fixed by the owners, we will remove it from our benchmark.

### 1.1. TFDS Datasets

Different streams are available, each is made of a sequence of datasets. When
iterating the datasets of a stream, TFDS datasets are automatically downloaded
on-the-fly if they don't exist.

### 1.2. Custom Dataset Downloaders

Many datasets implemented in Nevis can be automatically downloaded. This has to
be done in advance of training with the script `./build_dataset.sh`:

```bash
$ ./build_dataset.sh -h
Usage:
        -d <DATASET_NAME> | Dataset name
        -s <STREAM_NAME>  | Stream variant name (FULL|SHORT|TINY|DEBUG|...)
        -b                | Build docker before running
        -e                | Develop mode where code is mounted
        -h                | Help message
```

If running for the first time, pass the option `-b` alongside other commands to
build the **docker** (`nevis-data`). The develop mode is useful if you need to
change the codebase (e.g. for adding a new dataset) and need to debug quickly
without having to re-building the docker everytime (you still need to build the
docker in develop mode! `-b -e`).

See in `dm_nevis/streams/nevis_stream.py` the enum `NEVISStreamVariant` for the
full list of downloadable streams.

Some datasets are downloaded from Kaggle. See on the
[Kaggle website](https://www.kaggle.com/docs/api) how to configure your
credentials and place them in the folder `~/.kaggle`.

### 1.3. Manual Download

ImageNet is a TFDS Dataset but it needs to be downloaded manually. Please check
the [instructions](https://www.tensorflow.org/datasets/catalog/imagenet2012).

For info, TFDS will look for datasets in the directory defined by the
environment variable `TFDS_DATA_DIR`.

## 2. Experiments

Each experiment consists of training a model on a stream of multiple datasets.
Thus, this command will train a model on each dataset. We provide two main
paradigms of learners: *independent* and *finetune from previous*. In the
former, we create a new randomly initialized model for each dataset. In the
latter, a model is initialized for the first dataset of the stream, and tuned
sequentially for all datasets.

To launch an experiment, run:

```
./launch_local.sh <X> example
```

With `<X>` being the framework to use (`jax` or `torch`), second argument is the
config to use.

Note that for the torch version, if you want to run on gpu instead of cpu, you
need to provide the gpu id with `--device <GPU_ID>`. By default, the code is
using the id `-1` to symbolize cpu.

## Output directory for metrics

By default the metrics computed by `experiments_<X>/metrics/nevis_metrics.py`
will be written in `./nevis_output_dir`.

You can specify a different path by overriding the environment variable
`NEVIS_DEFAULT_OUTPUT_DIR`.

## Metrics visualization with TensorBoard

The TensorBoard events file will be saved to `~/nevis/tensorboard`. Each run
will create a folder below this directory named with the date and time when the
run was launched.

The tensorboard can be launched with the following command.

`tensorboard --lodir=~/nevis/tensorboard`

You will need to have `tensorboard` installed outside the docker using

```bash
pip install tensorboard
```

Regarding the different groups of plots on tensorboard dashboard: -
`benchmark_metrics` contains metrics from prediction events across the stream,
where the x-axis is the index (0-based) of the most training event. -
`train_event_<i>` contains training and validation metrics on the training index
with index `i`.

## 3. Example

Let's take an example learner (returns always zeros) that we will "train" on the
DEBUG stream made of Pascal VOC 2007 and Coil100 datasets.

Pascal VOC 2007 is a TFDS dataset so it will be automatically downloaded when
needed.

First we download Coil100 dataset:

```bash
./build_dataset.sh -e -b -s debug
```

Note that since the DEBUG stream only downloads Coil100, we could also have used
`-d coil100` instead of `-s debug`. As you can see in the script
`build_dataset.sh`, we download data in `~/nevis`. You can change this directory
by overriding the env variable `LOCAL_DIR` in the script.

Then, we launch the example learner:

```bash
./launch_local.sh jax example
```

Note that the stream `DEBUG` is already specified in the config
`./experiments_jax/config/example.py`.

## 4. Baselines

We provide several baselines, defined in the `learners/` directory with configurations
in the `configs/` directory. Note that the same approach might have multiple configurations.

Reminder, to run configuration `configs/X.py`, do `./launch_local.sh jax X.py`.

We provide the following baselines:
- **Independent**, in `configs/finetuning_ind.py` where each dataset is learned by an independent model
- **Previous**, in `configs/finetuning_prev.py` where we learn sequentially each dataset and initialize its parameters from the parameter vector learned on the previous task.
- **Dynamic**, in `configs/finetuning_dknn.py`. where the initialization of task T is chosen among the models which have been trained on a dataset most similar to the current dataset. This baseline performs hyperparameter tuning while learning the task, following the protocol described in our tech report.


Variants are also proposed, such as cheaper configurations in `configs/cheap_finetuning_dknn.py` which use a smaller net and fewer trials of hyper-parameter search. These are the best entry point for people who have access to only one or few GPUs.


It is also possible to run a pretrained model on the Nevis stream. First train 
your own pretrained model. For example on ImageNet, run the configuration `configs/pretrain_imagenet.py`. Collect the resulting checkpoint, see configuration file to see where it's saved. 
Then, use this checkpoint for `configs/finetuning_ind_pretrained.py`.

## 5. Code paths

The code is structured as follows:

```bash
|--- dm_nevis/
|    |--- benchmarker/
|    |--- datasets_storage/
|    |--- streams/
|--- experiments_jax/
|    |--- launch.py
|    |--- experiment.py
|    |--- configs/
|    |--- learners/
|    |--- metrics/
|    |--- environment/
|    |--- training/
|--- experiments_torch/
|    |--- launch.py
|    |--- experiment.py
|    |--- configs/
|    |--- learners/
|    |--- metrics/
|    |--- environment/
|    |--- training/
```

`dm_nevis/` is the library of the benchmark, containing the `benchmarker/`
library, which implements the evaluation protocol used in the [paper].
`datasets_storage/` is a package to support the downloading and preparation of
datasets, and `streams/` is a package defining different streams.

There are two directories containing baseline model implementations, one for jax
(`experiments_jax`), and one for pytorch (`experiments_torch`). In each,
`launch.py` is the Docker entrypoint, `experiment.py` is the module where all
the execution happens, `configs/` provides the hyperparameters for each learner,
`learners/` implements the learners (note: in some cases, there are different
configs for the same learner), `metrics/` implements the metrics used in
NEVIS'22, `environment/` provides the logger and checkpointer, and `training/`
provides learner-agnostic utilities such as the heads, the backbone, but also a
flops counter for example.

# Contact

If you wish to contact us, please raise a GitHub issue.

If you are using the NEVIS'22 benchmark, please cite the following paper,

```bibtex
@article{bornschein2022nevis,
  author={Bornschein, J\"org and Galashov, Alexandre and Hemsley, Ross and Rannen-Triki, Amal and Chen, Yutian and Chaudhry, Arslan and He, Xu Owen and Douillard, Arthur and Caccia, Massimo and Feng, Qixuang and Shen, Jiajun and Rebuffi, Sylvestre-Alvise and Stacpoole, Kitty and de las Casas, Diego and Hawkins, Will and Lazaridou, Angeliki and Teh, Yee Whye and Rusu, Andrei A. and Pascanu, Razvan and Ranzato, Marc'Aurelio},
  title={Nevis\'22: A Stream of 100 Tasks Sampled from 30 Years of Computer Vision Research},
  journal={CoRR},
  volume={abs/2211.11747},
  year={2022},
  url={https://arxiv.org/abs/2211.11747},
  eprinttype={arXiv}
}
```

[paper]: https://arxiv.org/abs/2211.11747
[blog post]: https://www.deepmind.com/blog/benchmarking-the-next-generation-of-never-ending-learners
[tfds]: https://www.tensorflow.org/datasets/api_docs/python/tfds
