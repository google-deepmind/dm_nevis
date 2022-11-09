# Datasets Storage

This is a package for generating sharded tfrecords datasets in a standardized
format, for a collection of datasets that are fetchable from the public
internet.

[I just want to know how to run the pipeline](#running-the-beam-pipeline) ↗️

## Handlers

The main component in this package are the *handlers*. Each handler provides the
necessary code to download the source *artifacts* for a dataset, and then
extract out the data into a unified format to be written as tfrecords.

### Single label handlers

Most common use-case for datasets is the standard classification. In this case,
a handler is expected to produce generators outputting a tuple of an image and
an integer label.

### Multi-label (and different cases) handlers

In different cases, the handler is expected to produce `handlers.types.Example`
protos. Moreover, a person writing handler must specify the `features` field in
the dataset metadata for it to be correctly parsed by the data pipeline.

## Dataset Artifacts

We define an *artifact* to be the raw downloadable data made available by a
dataset provider. Typically this might be a zip file, or other kind of
compressed file. These filetypes are convenient to distribute by dataset
providers, but do not provide a uniform way to read the contained data for
learning pipelines. The goal of the pipeline contained in this package is to
convert raw artifacts into uniform sharded tfrecords datasets.

Each dataset may have multiple artifacts, and these are specified by the
download links in the `DownloadableDataset` dataclass provided for each dataset
in the *handlers* package.

To ensure this data pipeline can be run reliably, we separate out the fetching
of artifacts for each dataset from the main pipeline, which converts the
artifacts into sharded tfrecords datasets.

The fetched dataset artifacts are stored on disk. This helps us ensure that we
can reliably re-run the data pipeline even if the raw source providers are down
temporarily, and also means that our data pipelines do not need access to the
public internet during execution.

For most datasets, it's possible to automatically download the dataset artifacts
from the internet using links. Some datasets, however must manually be
downloaded.

Some dataset *handlers* provide a way to write a fixture for their artifacts.
For these datasets, it's possible to write end-to-end tests.

See [main README.md](https://github.com/deepmind/dm_nevis/README.md) to download
and prepare the datasets.
