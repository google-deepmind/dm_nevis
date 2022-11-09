# Datasets and Tasks

The benchmarker supports a range of different datasets and tasks.
We outline the different task kinds below, along with their semantics.

## Task Kinds

### Single-label Image Classification

Let `x` be the input and `y` be the label. For this case, we consider `y` to
be a multinomial-distributed random variable taking values in
`{0, ..., num_classes - 1}`. Note that we adopt the same convention for the
special case of binary classification.

This task is selected using TaskKind.IMAGE_CLASSIFICATION, and in this case
there is a single, unique, integer label associated with each image example in
the dataset. This single integer is encoded into the minibatch object in the
label field. Predictions for this task are expected to be a discrete, normalized
multinomial probability distribution over the classes.

### Multi-Label Image Classification

Let `x` be the input and `y` be the label vector of shape `(num_classes)`.
`y_i` represents the presence of the `i`th class. For example, if
`y = (1, 1, 0, 0)`, this represents the event that the labels 0 and 1
are present, but the labels 3, 4 are not. Thus, each `y_i` represents
a Bernoulli-distributed random variable indicating the presence of class `i`,
encoded as a single value between 0 and 1.

In the TaskKind.MULTI_LABEL_IMAGE_CLASSIFICATION problem, each image is
associated with a sequence of `num_classes` Bernoulli random variables. The ith
random variable represents the estimated _marginal_ probability that the image
is associated with the ith label. The targets for this task are encoded as
a binary vector within the `datasets.MiniBatch` object in the `labels` field.

Predictions for this task take the form of a sequence of probabilities over
classes, similarly to the TaskKind.IMAGE_CLASSIFICATION task. However, it is no
longer required that the probabilities of each class sum to unity, since the
probability that a given image `X` has class `i` may be independent of the event
that the image `X` has class `j`. This independence is not a requirement,
however, and some datasets may exhibit a complicated join distribution across
labels. In particular the Pascal actions dataset introduces an `other` class,
which represents the event that none of the remaining labels are relevant to the
image in question. The dependency structure between the class random variables
is typically "hidden", and no guarantees are provided on this by the predictive
output.
