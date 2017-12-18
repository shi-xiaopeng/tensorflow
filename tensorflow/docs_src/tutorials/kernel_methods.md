# Improving Linear Models Using Explicit Kernel Methods
# 使用特定的内核方法改善线性模型

In this tutorial, we demonstrate how combining (explicit) kernel methods with
linear models can drastically increase the latters' quality of predictions
without significantly increasing training and inference times. Unlike dual
kernel methods, explicit (primal) kernel methods scale well with the size of the
training dataset both in terms of training/inference times and in terms of
memory requirements.

在这篇教程里，我们将示范如何把特定内核方法和线性模型结合起来在不明显增加训练和推理时间的情况下大幅度提升
预测质量。与多内核方法不同，特定核心方法对训练数据集的大小具有很好的可扩展性，不管是在训练和推理时间上
还是在内存占用上。

**Intended audience:** Even though we provide a high-level overview of concepts
related to explicit kernel methods, this tutorial primarily targets readers who
already have at least basic knowledge of kernel methods and Support Vector
Machines (SVMs). If you are new to kernel methods, refer to either of the
following sources for an introduction:

**目标读者：** 尽管我们提供的只是特定内核方法相关概念的一个概览，教程的主要目标读者
依然要具备内核方法和支持向量机的基础知识，这些知识可以参考以下的资源做一个了解：

* If you have a strong mathematical background:
[Kernel Methods in Machine Learning](https://arxiv.org/pdf/math/0701907.pdf)
* [Kernel method wikipedia page](https://en.wikipedia.org/wiki/Kernel_method)

* 如果你有很强的数学背景：
[机器学习中的内核方法](https://arxiv.org/pdf/math/0701907.pdf)
* [内核方法维基百科](https://en.wikipedia.org/wiki/Kernel_method)

Currently, TensorFlow supports explicit kernel mappings for dense features only;
TensorFlow will provide support for sparse features at a later release.

目前，TensorFlow 只支持对于密集特征的特定内核映射。对于稀疏特征的支持将在后面的发布版本中提供。

This tutorial uses [tf.contrib.learn](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn)
(TensorFlow's high-level Machine Learning API) Estimators for our ML models.
If you are not familiar with this API, [tf.estimator Quickstart](https://www.tensorflow.org/get_started/estimator)
is a good place to start. We will use the MNIST dataset. The tutorial consists
of the following steps:

本教程使用 [tf.contrib.learn](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn)
（TensorFlow 机器学习高级接口）估测器作为我们的机器学习模型。如果你对这份 API 不熟悉，可以参考
[tf.estimator Quickstart](https://www.tensorflow.org/get_started/estimator)。
我们将使用 MNIST 教程中提到数据集。本教程包含以下步骤：

* Load and prepare MNIST data for classification.
* Construct a simple linear model, train it, and evaluate it on the eval data.
* Replace the linear model with a kernelized linear model, re-train, and
re-evaluate.

* 为分类下载和准备 MNIST 数据。
* 构建一个简单的线性模型，训练并在评估数据上评估。
* 用内核化的线性模型替换线性模型，再次训练，评估。

## Load and prepare MNIST data for classification
## 为分类任务下载和准备 MNIST 数据

Run the following utility command to load the MNIST dataset:
运行如下的工具命令来下载 MNIST 数据集：

```python
data = tf.contrib.learn.datasets.mnist.load_mnist()
```
The preceding method loads the entire MNIST dataset (containing 70K samples) and
splits it into train, validation, and test data with 55K, 5K, and 10K samples
respectively. Each split contains one numpy array for images (with shape
[sample_size, 784]) and one for labels (with shape [sample_size, 1]). In this
tutorial, we only use the train and validation splits to train and evaluate our
models respectively.

上面的方法下载了整个 MNIST 数据集（包含 70K 的样本），并分成大小分别为 55K,
 5K 和 10K 的训练，验证和测试集。每次分割包含一个 numpy 图像数组
 （shape 为 [样本大小, 784]）和一个 numpy 标签数组（shape 为 [样本大小, 1])。
 在本教程中，我们仅使用训练和验证集来分别训练和验证我们的模型。

In order to feed data to a tf.contrib.learn Estimator, it is helpful to convert
it to Tensors. For this, we will use an `input function` which adds Ops to the
TensorFlow graph that, when executed, create mini-batches of Tensors to be used
downstream. For more background on input functions, check
@{$get_started/input_fn$Building Input Functions with tf.contrib.learn}. In this
example, we will use the `tf.train.shuffle_batch` Op which, besides converting
numpy arrays to Tensors, allows us to specify the batch_size and whether to
randomize the input every time the input_fn Ops are executed (randomization
typically expedites convergence during training). The full code for loading and
preparing the data is shown in the snippet below. In this example, we use
mini-batches of size 256 for training and the entire sample (5K entries) for
evaluation. Feel free to experiment with different batch sizes.

给 tf.contrib.learn 估测器输入数据时，把数据转换成张量会更方便。
为了做到这一点，我们使用一个`input function` 的方法，它会给 TensorFlow
计算图添加选项，执行时会创建下游使用的小批次张量。更多关于 `input function`
的背景消息请参考@{$get_started/input_fn$Building Input Functions with tf.contrib.learn}.
在本例中，我们将使用 `tf.train.shuffle_batch` 选项，这个选项除了可以将 numpy
数组转换张量之外，还允许我们指定每次 input_fn 选项运行时批次的大小以及是否将输入随机化
(随机化通常会加快训练过程中的收敛)。下载和准备数据的完整代码在下面的代码片段里。
例中，我们将训练中的子批次大小设置为 256，并使用整个的评估样本（5K）进行评估。
你也可以对其他的批次大小进行试验。

```python
import numpy as np
import tensorflow as tf

def get_input_fn(dataset_split, batch_size, capacity=10000, min_after_dequeue=3000):

  def _input_fn():
    images_batch, labels_batch = tf.train.shuffle_batch(
        tensors=[dataset_split.images, dataset_split.labels.astype(np.int32)],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True,
        num_threads=4)
    features_map = {'images': images_batch}
    return features_map, labels_batch

  return _input_fn

data = tf.contrib.learn.datasets.mnist.load_mnist()

train_input_fn = get_input_fn(data.train, batch_size=256)
eval_input_fn = get_input_fn(data.validation, batch_size=5000)

```

## Training a simple linear model
## 训练一个简单的线性模型

We can now train a linear model over the MNIST dataset. We will use the
@{tf.contrib.learn.LinearClassifier} estimator with 10 classes representing the
10 digits. The input features form a 784-dimensional dense vector which can
be specified as follows:

现在我们可以使用 MNIST 数据集训练一个线性模型。我们将使用 @{tf.contrib.learn.LinearClassifier}
估测器，10 个分类分别用 0-9 表示。输入特征组成一个 784 维的密集向量，可以通过如下方式指定：

```python
image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
```

The full code for constructing, training and evaluating a LinearClassifier
estimator is as follows:

构建、训练和评估一个线性分类器估测器的完整代码如下：

```python
import time

# Specify the feature(s) to be used by the estimator.
# 指定估测器要使用的特征
image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=10)

# Train.
# 训练
start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()
print('Elapsed time: {} seconds'.format(end - start))

# Evaluate and report metrics.
# 评估并报告结果
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)
```
The following table summarizes the results on the eval data.
下面这张表总结了评估数据的结果。

metric        | value
:------------ | :------------
loss          | 0.25 to 0.30
accuracy      | 92.5%
training time | ~25 seconds on my machine

Note: Metrics will vary depending on various factors.
注意：评估结果会因各种因素的影响而不同。

In addition to experimenting with the (training) batch size and the number of
training steps, there are a couple other parameters that can be tuned as well.
For instance, you can change the optimization method used to minimize the loss
by explicitly selecting another optimizer from the collection of
[available optimizers](https://www.tensorflow.org/code/tensorflow/python/training).
As an example, the following code constructs a LinearClassifier estimator that
uses the Follow-The-Regularized-Leader (FTRL) optimization strategy with a
specific learning rate and L2-regularization.

除了可以试验不同的批次大小和训练步骤的数量，有一些其他的因素同样可以被调优。
例如，你可以通过从[可用优化器列表](https://www.tensorflow.org/code/tensorflow/python/training)
明确地选择另外的优化器，改变用于最小化损失函数的优化方法来优化评估结果。作为示例，下面的代码构造了一个使用
特定学习速率和 L2 正规化的FTRL(Follow-The-Regularized-Leader) 优化策略的线性分类估测器。

```python
optimizer = tf.train.FtrlOptimizer(learning_rate=5.0, l2_regularization_strength=1.0)
estimator = tf.contrib.learn.LinearClassifier(
    feature_columns=[image_column], n_classes=10, optimizer=optimizer)
```

Regardless of the values of the parameters, the maximum accuracy a linear model
can achieve on this dataset caps at around **93%**.
不管这些参数的值，线性模型在这个数据集上能取得的最大精确度约为**93%**。


## Using explicit kernel mappings with the linear model.
## 在线性模型上使用特定内核映射

The relatively high error (~7%) of the linear model over MNIST indicates that
the input data is not linearly separable. We will use explicit kernel mappings
to reduce the classification error.

线性模型在 MNIST 数据上的相对较高的错误率表明输入数据不是可线性分割的。我们将使用特定的内核
映射来降低分类误差。

**Intuition:** The high-level idea is to use a non-linear map to transform the
input space to another feature space (of possibly higher dimension) where the
(transformed) features are (almost) linearly separable and then apply a linear
model on the mapped features. This is shown in the following figure:

**直觉上：** 总体思路是用一个非线性的映射关系将输入空间转换到另一个特征空间（可能有更高的维数），
这个特征空间几乎是可线性分割的，然后把一个线性模型应用到这个映射特征空间里。这个过程可以通过
下面这张图反映：

<div style="text-align:center">
<img src="https://www.tensorflow.org/versions/master/images/kernel_mapping.png" />
</div>


### Technical details
### 技术细节

In this example we will use **Random Fourier Features**, introduced in the
["Random Features for Large-Scale Kernel Machines"](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
paper by Rahimi and Recht, to map the input data. Random Fourier Features map a
vector \\(\mathbf{x} \in \mathbb{R}^d\\) to \\(\mathbf{x'} \in \mathbb{R}^D\\)
via the following mapping:

这个例子中我们将使用 **随机傅里叶特征**，在 Rahimi 和 Recht 的对输入数据做映射的论文中首次引进，
["大规模内核机器的随机特征"](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)。
随机傅里叶特征通过如下方式将矢量 \\(\mathbf{x} \in \mathbb{R}^d\\)
映射到 \\(\mathbf{x'} \in \mathbb{R}^D\\)：

$$
RFFM(\cdot): \mathbb{R}^d \to \mathbb{R}^D, \quad
RFFM(\mathbf{x}) =  \cos(\mathbf{\Omega} \cdot \mathbf{x}+ \mathbf{b})
$$

where \\(\mathbf{\Omega} \in \mathbb{R}^{D \times d}\\),
\\(\mathbf{x} \in \mathbb{R}^d,\\) \\(\mathbf{b} \in \mathbb{R}^D\\) and the
cosine is applied element-wise.

In this example, the entries of \\(\mathbf{\Omega}\\) and \\(\mathbf{b}\\) are
sampled from distributions such that the mapping satisfies the following
property:

在本例中，\\(\mathbf{\Omega}\\) 和 \\(\mathbf{b}\\) 的实体是分布中取样得来的以使
映射具有如下性质：

$$
RFFM(\mathbf{x})^T \cdot RFFM(\mathbf{y}) \approx
e^{-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2 \sigma^2}}
$$

The right-hand-side quantity of the expression above is known as the RBF (or
Gaussian) kernel function. This function is one of the most-widely used kernel
functions in Machine Learning and implicitly measures similarity in a different,
much higher dimensional space than the original one. See
[Radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
for more details.

上面表达式右边的式子就是著名的 RBF （高斯）内核函数。这个函数是机器学习中应用最广泛的内核函数之一，
相比原始函数它可以在不同和更高的维度空间中模糊地衡量相似性。

### Kernel classifier
### 内核分类器

@{tf.contrib.kernel_methods.KernelLinearClassifier} is a pre-packaged
`tf.contrib.learn` estimator that combines the power of explicit kernel mappings
with linear models. Its constructor is almost identical to that of the
LinearClassifier estimator with the additional option to specify a list of
explicit kernel mappings to be applied to each feature the classifier uses. The
following code snippet demonstrates how to replace LinearClassifier with
KernelLinearClassifier.

@{tf.contrib.kernel_methods.KernelLinearClassifier} 是一个
预先打包的 `tf.contrib.learn` 估测器，结合了线性模型和特定内核映射的优点。
它的构造器几乎与可配置内核映射的线性分类估测器相同。
下面的代码演示如何使用线性内核分类器替换线性分类器。

```python
# Specify the feature(s) to be used by the estimator. This is identical to the
# code used for the LinearClassifier.
# 指定估测器将要使用的特征。这和线性分类器中的代码是相同的。

image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
optimizer = tf.train.FtrlOptimizer(
   learning_rate=50.0, l2_regularization_strength=0.001)


kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
  input_dim=784, output_dim=2000, stddev=5.0, name='rffm')
kernel_mappers = {image_column: [kernel_mapper]}
estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
   n_classes=10, optimizer=optimizer, kernel_mappers=kernel_mappers)

# Train.
# 训练。
start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()
print('Elapsed time: {} seconds'.format(end - start))

# Evaluate and report metrics.
# 评估和报告结果。
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)
```
The only additional parameter passed to `KernelLinearClassifier` is a dictionary
from feature_columns to a list of kernel mappings to be applied to the
corresponding feature column. The following lines instruct the classifier to
first map the initial 784-dimensional images to 2000-dimensional vectors using
random Fourier features and then learn a linear model on the transformed
vectors:

唯一增加的一个参数是传递给 `KernelLinearClassifier` 的一个字典，这个字典提供
从特征列到一组内核的映射。下面的几行构造了使用随机傅立叶特征来映射初始化的 784 维图像到一个
2000 维的矢量的一个分类器，然后在经过转换的矢量上进行学习。

```python
kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
  input_dim=784, output_dim=2000, stddev=5.0, name='rffm')
kernel_mappers = {image_column: [kernel_mapper]}
estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
   n_classes=10, optimizer=optimizer, kernel_mappers=kernel_mappers)
```
Notice the `stddev` parameter. This is the standard deviation (\\(\sigma\\)) of
the approximated RBF kernel and controls the similarity measure used in
classification. `stddev` is typically determined via hyperparameter tuning.

注意 `stddev` 参数。它是 RBF 内核近似的标准差并在分类任务里用于控制相似性衡量。
`stddev` 通常通过超参数调教决定。

The results of running the preceding code are summarized in the following table.
We can further increase the accuracy by increasing the output dimension of the
mapping and tuning the standard deviation.

先前代码的运行结果总结在下面这张表里。我们可以通过提高映射输出维数和对标准差调优
进一步提高准确性。

metric        | value
:------------ | :------------
loss          | 0.10
accuracy      | 97%
training time | ~35 seconds on my machine


### stddev

The classification quality is very sensitive to the value of stddev. The
following table shows the accuracy of the classifier on the eval data for
different values of stddev. The optimal value is stddev=5.0. Notice how too
small or too high stddev values can dramatically decrease the accuracy of the
classification.

分类的质量对 stddev 的值非常敏感。下面这张表显示的是分类器在不同 stddev 上的
评估数据的准确性。最优值是 stddev=5.0。注意看太大或者太小的 stddev 值是怎样迅速的
降低分类的准确性的。

stddev | eval accuracy
:----- | :------------
1.0    | 0.1362
2.0    | 0.4764
4.0    | 0.9654
5.0    | 0.9766
8.0    | 0.9714
16.0   | 0.8878

### Output dimension
### 输出维数

Intuitively, the larger the output dimension of the mapping, the closer the
inner product of two mapped vectors approximates the kernel, which typically
translates to better classification accuracy. Another way to think about this is
that the output dimension equals the number of weights of the linear model; the
larger this dimension, the larger the "degrees of freedom" of the model.
However, after a certain threshold, higher output dimensions increase the
accuracy by very little, while making training take more time. This is shown in
the following two Figures which depict the eval accuracy as a function of the
output dimension and the training time, respectively.

直觉上，映射的输出维数越大，两个映射的矢量的内积越近似这个内核，通常都会转化成更好的分类准确性。
另一个思考这个问题的方式是输出维数等于线性模型的权重数；维数越大，模型的自由度越大。
然而，超过一定的阈值，随着训练次数的增加，高输出维数对准确性的提升作用越来越小。下面
两张图分别描述了评价准确性对输出维数和训练时间的函数变化

![image](https://www.tensorflow.org/versions/master/images/acc_vs_outdim.png)
![image](https://www.tensorflow.org/versions/master/images/acc-vs-trn_time.png)


## Summary
## 总结
Explicit kernel mappings combine the predictive power of nonlinear models with
the scalability of linear models. Unlike traditional dual kernel methods,
explicit kernel methods can scale to millions or hundreds of millions of
samples. When using explicit kernel mappings, consider the following tips:

特定内核映射结合了非线性模型的前瞻性和线性模型的可伸缩性的优点。与传统的多核心方法相比，
特定内核方法可以覆盖几十万到几千万量级的实例。当运用特定内核映射时，请考虑如下几点：

* Random Fourier Features can be particularly effective for datasets with dense
features.
* The parameters of the kernel mapping are often data-dependent. Model quality
can be very sensitive to these parameters. Use hyperparameter tuning to find the
optimal values.
* If you have multiple numerical features, concatenate them into a single
multi-dimensional feature and apply the kernel mapping to the concatenated
vector.

* 对于密集特征数据集，随机傅立叶特征特别有效
* 内核映射的参数经常是独立于数据的。模型质量对于这些参数非常敏感。使用超参数调优找到最优值。
* 如果有多个数字特征，把他们连接到一个单独的多维特征中然后在这些连接数据上运用内核映射。
