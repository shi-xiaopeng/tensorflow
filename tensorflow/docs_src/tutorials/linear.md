# Large-scale Linear Models with TensorFlow
# TensorFlow 的大规模线性模型

The tf.estimator API provides (among other things) a rich set of tools for
working with linear models in TensorFlow. This document provides an overview of
those tools. It explains:

tf.estimator 的 API (和其他工具一起）已经为在 TensorFlow 中使用线性模型提供了一系列丰富的工具。
这个文档将对这些工具做一个概览。 它包括：

  
   * what a linear model is.
   * why you might want to use a linear model.
   * how tf.estimator makes it easy to build linear models in TensorFlow.
   * how you can use tf.estimator to combine linear models with
   deep learning to get the advantages of both.

   * 线性模型是什么。
   * 为什么要使用线性模型。
   * tf.estimator 是如何使线性模型的构建更简单的。
   * 怎样使用 tf.estimator 融合线性模型和深度学习更好的发挥两者的优势
   
Read this overview to decide whether the tf.estimator linear model tools might
be useful to you. Then do the @{$wide$Linear Models tutorial} to
give it a try. This overview uses code samples from the tutorial, but the
tutorial walks through the code in greater detail.

你可以通过这个概览知道 tf.estimator 的线性模型工具是否对你有帮助。而后你可以再 @{$wide$线性模型教程}
中尝试一下。这个概览的代码用例来自于教程，但是教程会对代码有更详细的说明。

To understand this overview it will help to have some familiarity
with basic machine learning concepts, and also with @{$estimator$tf.estimator}.

为了更好的理解这个概览，你应该首先对机器学习的基本概念和 @{$estimator$tf.estimator} 有所了解。

[TOC]

## What is a linear model?
## 线性模型是什么？

A *linear model* uses a single weighted sum of features to make a prediction.
For example, if you have [data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)
on age, years of education, and weekly hours of
work for a population, you can learn weights for each of those numbers so that
their weighted sum estimates a person's salary. You can also use linear models
for classification.

*线性模型*使用单一变量对所有的特征做出预测。例如，如果你有有关年龄的数据，受教育年限，每周的工作时长，
你可以从这些数据中学习到权重值，使得这个权重乘以总值可以预测出一个人的薪水。你同样可以用线性模型来做分类。

Some linear models transform the weighted sum into a more convenient form. For
example, *logistic regression* plugs the weighted sum into the logistic
function to turn the output into a value between 0 and 1. But you still just
have one weight for each input feature.

一些线性模型把这个加权和转换成为一种更简便的形式。例如，逻辑回归将加权和导入一个逻辑函数中，
获得一个在 0 和 1 之间的输出。但是对于每次输入的特征依然只有一个权重值。

## Why would you want to use a linear model?

## 为什么要使用线性模型

Why would you want to use so simple a model when recent research has
demonstrated the power of more complex neural networks with many layers?

在当前研究显示出更复杂的多层神经网络的威力的情况下，我们为什么还有使用如此简单的线性模型呢？

Linear models:
线性模型：

   * train quickly, compared to deep neural nets.
   * can work well on very large feature sets.
   * can be trained with algorithms that don't require a lot of fiddling
   with learning rates, etc.
   * can be interpreted and debugged more easily than neural nets.
   You can examine the weights assigned to each feature to figure out what's
   having the biggest impact on a prediction.
   * provide an excellent starting point for learning about machine learning.
   * are widely used in industry.

   * 相对于深度神经网络，线性模型的训练速度更快。
   * 在非常巨大的特征集上依然有效。
   * 能够使用不需要很多无用的学习率的算法进行训练。
   * 比神经网络更容易理解和调试。你可以查看分配给每一个特征的权重值来搞清楚什么会对预测产生最大的影响。
   * 是学习机器学习的一个绝佳的起始点。
   * 在工业中普遍使用。

## How does tf.estimator help you build linear models?

## tf.estimator 是如何帮助你构建线性模型的？

You can build a linear model from scratch in TensorFlow without the help of a
special API. But tf.estimator provides some tools that make it easier to build
effective large-scale linear models.

在 TensorFlow 中你可以不借助于任何特殊的 API 来从头创建一个线性模型。但是 tf.estimator
提供了一些工具使构建有效的大规模线性模型更容易。

### Feature columns and transformations

### 特征列和转换

Much of the work of designing a linear model consists of transforming raw data
into suitable input features. Tensorflow uses the `FeatureColumn` abstraction to
enable these transformations.

设计一个线性模型的大部分工作集中在把原始数据转换成合适的输入特征。TensorFlow 使用
`特征列` 的抽象方式使这些转换成为可能。

A `FeatureColumn` represents a single feature in your data. A `FeatureColumn`
may represent a quantity like 'height', or it may represent a category like
'eye_color' where the value is drawn from a set of discrete possibilities like
{'blue', 'brown', 'green'}.

一个 `特征列` 表示你的数据中的一个单一特征。一个 `特征列` 可能表示一个数量，如高度；
也可能代表一中分类，如眼睛的颜色，眼睛的颜色可能是一系列可能的颜色如 {'蓝', '棕', '绿'}。

In the case of both *continuous features* like 'height' and *categorical
features* like 'eye_color', a single value in the data might get transformed
into a sequence of numbers before it is input into the model. The
`FeatureColumn` abstraction lets you manipulate the feature as a single
semantic unit in spite of this fact. You can specify transformations and
select features to include without dealing with specific indices in the
tensors you feed into the model.

不管是连续性特征（如身高）还是类别性特征（如眼睛颜色），数据中的一个单一值在输入模型之前
都可能会被转换成一个数值序列。抽象的 `特征列` 使你能像操作单个语义单元一样对特征进行操作。
你可以指定进行那种转换，选择要加入的特征而不用担心模型输入张量的特定索引。

#### Sparse columns

### 稀疏列

Categorical features in linear models are typically translated into a sparse
vector in which each possible value has a corresponding index or id. For
example, if there are only three possible eye colors you can represent
'eye_color' as a length 3 vector: 'brown' would become [1, 0, 0], 'blue' would
become [0, 1, 0] and 'green' would become [0, 0, 1]. These vectors are called
"sparse" because they may be very long, with many zeros, when the set of
possible values is very large (such as all English words).

线性模型中的分类特征通常会被转换到一个稀疏向量中，向量中的每个可能值都有相应的 id 或索引。
例如，如果只有三种可能的眼睛颜色，你可以使用一个长度为 3 的向量来表示：[1, 0, 0] 表示 '棕'，
[0, 1, 0] 表示 '蓝'，[0, 0, 1] 表示 '绿'。这些向量之所以叫 '稀疏' 是因为当可能的
取值非常大的时候（例如所有的英文单词），向量就会就会非常长而且会有很多 0.

While you don't need to use categorical columns to use tf.estimator linear
models, one of the strengths of linear models is their ability to deal with
large sparse vectors. Sparse features are a primary use case for the
tf.estimator linear model tools.

当你使用 tf.estimator 的线性模型时，就不在需要用分类列了。线性模型的优点之一就是他们
处理大型稀疏向量的能力。稀疏特征是 tf.estimator 线性特征模型工具的一个主要使用场景。


##### Encoding sparse columns
##### 编码稀疏列

`FeatureColumn` handles the conversion of categorical values into vectors
automatically, with code like this:

 `特征列` 会自动处理分类值到向量的转换过程，使用一下的代码：

```python
eye_color = tf.feature_column.categorical_column_with_vocabulary_list(
    "eye_color", vocabulary_list=["blue", "brown", "green"])
```

where `eye_color` is the name of a column in your source data.
这里的 `eye_color` 就是你的原始数据中列的名字。

You can also generate `FeatureColumn`s for categorical features for which you
don't know all possible values. For this case you would use
`categorical_column_with_hash_bucket()`, which uses a hash function to assign
indices to feature values.

你同样可以为你不知道所有的肯能取值的分类特征生成 `特征列`。这种情况下你应该使用
`categorical_column_with_hash_bucket()`，这个方法会使用哈希函数为特征值建立索引。

```python
education = tf.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
```

##### Feature Crosses
##### 交叉特征

Because linear models assign independent weights to separate features, they
can't learn the relative importance of specific combinations of feature
values. If you have a feature 'favorite_sport' and a feature 'home_city' and
you're trying to predict whether a person likes to wear red, your linear model
won't be able to learn that baseball fans from St. Louis especially like to
wear red.

由于线性模型会为不同的特征分配单独的权重，线性模型无法学习特定的特征组合的相对重要性。
如果你有 "最喜欢的运动" 和 "家乡城市" 这两个特征，然后尝试预测是否一个人喜欢穿红色，你的
线性模型时没有办法学到来自圣路易斯的棒球迷特别喜欢穿红色的。

You can get around this limitation by creating a new feature
'favorite_sport_x_home_city'. The value of this feature for a given person is
just the concatenation of the values of the two source features:
'baseball_x_stlouis', for example. This sort of combination feature is called
a *feature cross*.

你可以通过创建一个表示 "最喜欢的运动-家乡城市" 的新特征来绕开这个限制。对于给定的一个人
这个特征的值正好是那两个源特征的值的连接：例如，"棒球-圣路易斯"。这种结合特征被叫做
"交叉特征"。

The `crossed_column()` method makes it easy to set up feature crosses:
`crossed_column()` 方法使设置交叉特征很容易：

```python
sport_x_city = tf.feature_column.crossed_column(
    ["sport", "city"], hash_bucket_size=int(1e4))
```

#### Continuous columns
#### 连续性列

You can specify a continuous feature like so:
你能像下面这样指定一个连续性特征：

```python
age = tf.feature_column.numeric_column("age")
```

Although, as a single real number, a continuous feature can often be input
directly into the model, Tensorflow offers useful transformations for this sort
of column as well.
尽管作为一个单一的实数，通常情况下连续性特征能直接被输入模型中，TensorFlow 同样为
这种特征列提供了很有用的转换方式。


##### Bucketization
##### 离散化

*Bucketization* turns a continuous column into a categorical column. This
transformation lets you use continuous features in feature crosses, or learn
cases where specific value ranges have particular importance.

*离散化* 能把连续性列转换成分类性列。这种转换使你能在交叉特征中使用连续性列，或者从
特定的区间有特定的权重的特征列中学习

Bucketization divides the range of possible values into subranges called
buckets:

离散化把可能值的区间划分成一个个小区间，这些小的区间就叫作 ucket 。

```python
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

The bucket into which a value falls becomes the categorical label for
that value.

一个值所落入的 bucket 就成为这个值得分类标签。


#### Input function
#### 输入函数

`FeatureColumn`s provide a specification for the input data for your model,
indicating how to represent and transform the data. But they do not provide
the data itself. You provide the data through an input function.

`特征列` 为模型提供了一种输入数据规格，指明如何表示和转换数据。但是它们本身不是数据。
你需要通过一个输入函数提供数据。

The input function must return a dictionary of tensors. Each key corresponds to
the name of a `FeatureColumn`. Each key's value is a tensor containing the
values of that feature for all data instances. See
@{$input_fn$Building Input Functions with tf.estimator} for a
more comprehensive look at input functions, and `input_fn` in the
[linear models tutorial code](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)
for an example implementation of an input function.

这个输入函数必须返还一个张量字典。其中的每一个键对应 `特征列` 的名字，键所对应的值是一个张量，
包含所有数据实例的该特征的值。想要更多的了解输入函数请看@{$input_fn$使用 tf.estimator 构建输入函数}，
一个输入函数的实现例子请见：
[线性模型教程代码](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)

The input function is passed to the `train()` and `evaluate()` calls that
initiate training and testing, as described in the next section.

输入函数在调用 `train()` 和 `evaluate()` 初始化训练和测试时被传进来，将在下一部分说明。

### Linear estimators
### 线性估算器

Tensorflow estimator classes provide a unified training and evaluation harness
for regression and classification models. They take care of the details of the
training and evaluation loops and allow the user to focus on model inputs and
architecture.

Tensorflow 估算器类为回归和分类模型提供一套统一的训练和评估框架。它们会处理好
训练和评估循环过程中的细节，让用户专注于模型的输入和结构。

To build a linear estimator, you can use either the
`tf.estimator.LinearClassifier` estimator or the
`tf.estimator.LinearRegressor` estimator, for classification and
regression respectively.
你可以使用 `tf.estimator.LinearClassifier` `tf.estimator.LinearRegressor`
来创建估算器分别用于分类和回归。

As with all tensorflow estimators, to run the estimator you just:
对于所有的 tensorflow 估算器，运行它只需要：

   1. Instantiate the estimator class. For the two linear estimator classes,
   you pass a list of `FeatureColumn`s to the constructor.
   2. Call the estimator's `train()` method to train it.
   3. Call the estimator's `evaluate()` method to see how it does.

   1. 初始化估算器。对于线性估算器类，你为构造器传入一个`特征列`列表。
   2. 调用估算器的 `train()` 方法训练它。
   3. 调用估算器的 `evaluate()` 方法查看训练的效果。

For example:
例如：

```python
e = tf.estimator.LinearClassifier(
    feature_columns=[
        native_country, education, occupation, workclass, marital_status,
        race, age_buckets, education_x_occupation,
        age_buckets_x_race_x_occupation],
    model_dir=YOUR_MODEL_DIRECTORY)
e.train(input_fn=input_fn_train, steps=200)
# Evaluate for one step (one pass through the test data).
# 对训练进行评估（通过测试数据）
results = e.evaluate(input_fn=input_fn_test)

# Print the stats for the evaluation.
# 打印评估的结果数据
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
```

### Wide and deep learning
### 宽深学习
The tf.estimator API also provides an estimator class that lets you jointly
train a linear model and a deep neural network. This novel approach combines the
ability of linear models to "memorize" key features with the generalization
ability of neural nets. Use `tf.estimator.DNNLinearCombinedClassifier` to
create this sort of "wide and deep" model:

tf.estimator API 同样提供了一个估算器类能让你同时训练一个线性模型和一个深度神经网络。
这个新颖的方法结合了线性模型对关键特征的记忆和神经网络的通用性能力。可以使用
`tf.estimator.DNNLinearCombinedClassifier` 创建这种宽深模型

```python
e = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=YOUR_MODEL_DIR,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
```
For more information, see the @{$wide_and_deep$Wide and Deep Learning tutorial}.
更多信息，请见 @{$wide_and_deep$宽深学习教程}.
