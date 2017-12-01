# How to Retrain Inception's Final Layer for New Categories
# 如何再训练 Inception 的最后一层识别新分类
Modern object recognition models have millions of parameters and can take weeks
to fully train. Transfer learning is a technique that shortcuts a lot of this
work by taking a fully-trained model for a set of categories like ImageNet, and
retrains from the existing weights for new classes. In this example we'll be
retraining the final layer from scratch, while leaving all the others untouched.
For more information on the approach you can see
[this paper on Decaf](https://arxiv.org/pdf/1310.1531v1.pdf).

当前的对象识别模型拥有数十万计的参数，花费数周的时间来整个训练。迁移学习是一种能够大幅缩短这一过程的一种技术，
它通过将一个已经完整训练过的模型如 ImageNet 重新训练来识别新的分类。
在本例中我们将从零开始训练模型的最后一层并保留其他部分不变。
想要获得此方法的更多信息请参考[ Decaf 的这篇论文](https://arxiv.org/pdf/1310.1531v1.pdf).

Though it's not as good as a full training run, this is surprisingly effective
for many applications, and can be run in as little as thirty minutes on a
laptop, without requiring a GPU. This tutorial will show you how to run the
example script on your own images, and will explain some of the options you have
to help control the training process.

尽管这种方式的效果没有比不上完整的训练，这种方法却对很多应用惊人的有效，而且能在笔记本上不要求 GPU的情况下 30 分钟内完成训练。
这篇教程将展示如何在你自己的图片库上执行示例脚本，而且会讲解一些你将会用到的，帮助控制训练过程的一些选项。

Note: This version of the tutorial mainly uses bazel. A bazel free version is
also available
[as a codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0).

注：此版本的教程主要使用 Bazel 构建工具。下面给出一个 codelab 上的一个免费版本
(https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0).

[TOC]

## Training on Flowers
## 训练对花的识别

![Daisies by Kelly Sikkema](https://www.tensorflow.org/images/daisies.jpg)

[Image by Kelly Sikkema](https://www.flickr.com/photos/95072945@N05/9922116524/)

[Kelly Sikkema 提供](https://www.flickr.com/photos/95072945@N05/9922116524/)

Before you start any training, you'll need a set of images to teach the network
about the new classes you want to recognize. There's a later section that
explains how to prepare your own images, but to make it easy we've created an
archive of creative-commons licensed flower photos to use initially. To get the
set of flower photos, run these commands:

在开始任何的训练之前，你需要一组图片，用于教网络认识你想让网络识别的那些新分类。
具体如何准备你自己的图片库我们将在后面的部分讲解，为了便于讲解我们创建了一个
包含知识共享授权花的图片的文件夹用于我们的初始化。
获取这些花朵图片，可以执行下面的命令：

```sh
cd ~
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
```

Once you have the images, you can build the retrainer like this, from the root
of your TensorFlow source directory:

获得这些图片后，你可以在你的 TensorFlow 源码文件的根目录下构建这个再训练器 :

```sh
bazel build tensorflow/examples/image_retraining:retrain
```

If you have a machine which supports
[the AVX instruction set](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
(common in x86 CPUs produced in the last few years) you can improve the running
speed of the retraining by building for that architecture, like this (after choosing appropriate options in `configure`):

如果你有一个支持[AVX 指令集](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)(常见于最近几年中生产的 x86 CPU 中)
的机器，你可以通过构建能充分利用那种架构优点的体系结构来提升再训练的执行速度，构建方式如下（在 `configure` 中选择合适的选项之后）:

```sh
bazel build --config opt tensorflow/examples/image_retraining:retrain
```

The retrainer can then be run like this:
之后再训练器可以按照如下方式运行：

```sh
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/flower_photos
```

This script loads the pre-trained Inception v3 model, removes the old top layer,
and trains a new one on the flower photos you've downloaded. None of the flower
species were in the original ImageNet classes the full network was trained on.
The magic of transfer learning is that lower layers that have been trained to
distinguish between some objects can be reused for many recognition tasks
without any alteration.

这行代码加载了先前训练的 Inception v3 模型，移去最顶层，在你下载的花朵图片上训练生成新的最顶层。
先前完整训练的生成的原始 ImageNet 分类中不存在任何一种花的类型。迁移学习的神奇之处就在于
之前的训练已经使识别网络的低层级能够识别不同的对象，可以在其他很多识别任务中重用而不用做任何修改。


## Bottlenecks
## 瓶颈

The script can take thirty minutes or more to complete, depending on the speed
of your machine.
根据你的机器速度的不同，这个脚本可能需要 30 分钟或更长时间才能完成，
The first phase analyzes all the images on disk and calculates
the bottleneck values for each of them. 'Bottleneck' is an informal term we
often use for the layer just before the final output layer that actually does
the classification.
第一个阶段会分析磁盘上的所有图片并计算出每一个的 Bottleneck 值。'Bottleneck' 是一个非正式词汇，
我们经常用它指代最后的输出层也就是最终做分类的那一层的前一层。

This penultimate layer has been trained to output a set of
values that's good enough for the classifier to use to distinguish between all
the classes it's been asked to recognize. 
这个倒数第二层已经被训练并能够输出一组值，这组值已经好到可以让分类器依据这些值来完成它的分类任务。
That means it has to be a meaningful
and compact summary of the images, since it has to contain enough information
for the classifier to make a good choice in a very small set of values. 
这意味着这些值必须是这些图片的精简而有意义的总结，因为它要包含足够的信息
来让分类器在一个很小的数值区间中做出好的选择。
The reason our final layer retraining can work on new classes is that it turns out
the kind of information needed to distinguish between all the 1,000 classes in
ImageNet is often also useful to distinguish between new kinds of objects.
最后一层的重新训练能够识别新分类的原因是，用于分辨 1000 中分类的信息对于识别新分类通常也时分有用。

Because every image is reused multiple times during training and calculating
each bottleneck takes a significant amount of time, it speeds things up to
cache these bottleneck values on disk so they don't have to be repeatedly
recalculated. By default they're stored in the `/tmp/bottleneck` directory, and
if you rerun the script they'll be reused so you don't have to wait for this
part again.
由于在训练和计算 bottleneck 值时每一图片都会被多次使用，因此把计算过的 bottleneck 值缓存在磁盘中
会大幅提升训练的速度，因为不用再重复计算了。bottleneck 值默认保存在 `/tmp/bottleneck` 目录下，
如果重复执行脚本他们会被重用，因此训练用时会比第一次训练要短。

## Training
## 训练

Once the bottlenecks are complete, the actual training of the top layer of the
network begins. 
bottleneck 值计算完成后，网络的顶层训练才真正开始。
You'll see a series of step outputs, each one showing training
accuracy, validation accuracy, and the cross entropy. 
你将看到一系列步骤的输出，每一天都会显示训练的精确度、验证的精确度以及交叉熵。
The training accuracy shows what percent of the images used in the current training batch were
labeled with the correct class. 
训练准确度显示当前训练所用的图片被正确分类的百分比。
The validation accuracy is the precision on a randomly-selected group of images from a different set. 
验证的准确度指从另一个集合中随机选出的一组图片被正确分类的百分比。
The key difference is that the training accuracy is based on images that the network has been able
to learn from so the network can overfit to the noise in the training data.
上面两个指标的关键差别在于，训练正确度的计算是依据网络已经学习过的那些图片，因此网络能够对训练数据中的噪声过拟合。
A true measure of the performance of the network is to measure its performance on
a data set not contained in the training data -- this is measured by the
validation accuracy. 
衡量一个网络表现的真正标准应该是它在训练数据集之外的数据上的表现 -- 通过验证正确性这个指标来衡量。
If the train accuracy is high but the validation accuracy
remains low, that means the network is overfitting and memorizing particular
features in the training images that aren't helpful more generally. 
如果训练准确定很高，验证准确定却很低，说明网络过拟合了，它已经记住了训练图片的特有的特征，这无助于更加通用的识别。
Cross entropy is a loss function which gives a glimpse into how well the learning
process is progressing. 
交叉熵是一个可以让我们一瞥学习进行情况的损失函数。
The training's objective is to make the loss as small as
possible, so you can tell if the learning is working by keeping an eye on
whether the loss keeps trending downwards, ignoring the short-term noise.
训练的目标是让这个损失函数的值尽可能的小，所以通过观察忽略短期噪声情况下
损失函数是否趋势递减来判断学习过程的进行状况。
By default this script will run 4,000 training steps. Each step chooses ten
images at random from the training set, finds their bottlenecks from the cache,
and feeds them into the final layer to get predictions. 
默认情况下，这个脚本会执行 4000 次训练。每次训练会从训练集中随机选取 10 张图片，从缓存中找到
他们的 bottleneck 值，把他们传给最后一层来获得预测结果。
Those predictions are then compared against the actual labels to update the final layer's weights
through the back-propagation process. 
这些预测值随后会与真实的标签值进行比较，并通过反向传播过程对优化最后一层的权重分布。
As the process continues you should see
the reported accuracy improve, and after all the steps are done, a final test
accuracy evaluation is run on a set of images kept separate from the training
and validation pictures.
随着这一过程的进行你会看到报告的准确性在提升，当所有的训练步骤完成后，
会在一个独立于测试集和验证集的图片集合上进行最终的测试准确性评估。
This test evaluation is the best estimate of how the
trained model will perform on the classification task.
这个测试评估是对这个训练模型在分类任务中表现如何的最好估测。
You should see an accuracy value of between 90% and 95%, though the exact value will vary from run
to run since there's randomness in the training process.
你会看到一个介于 90% 和 95% 的准确性值，尽管每一次运行的确切值都不同，因为训练过程有随机性存在。
This number is based on the percent of the images in the test set that are given the correct label
after the model is fully trained.
这个值是模型训练完成后对测试图片集正确分类的比例。

## Visualizing the Retraining with TensorBoard
## 使用 TensorBoard 可视化再训练过程

The script includes TensorBoard summaries that make it easier to understand, debug, and optimize the retraining. For example, you can visualize the graph and statistics, such as how the weights or accuracy varied during training.

这些脚本包括 TensorBoard 的简要介绍，使得理解，调试和优化过程更易理解。例如，你可以可视化诸如训练过程中的权重和准确性变化的图表和统计数据。

To launch TensorBoard, run this command during or after retraining:
要启动 TensorBoard 请在训练过程中或训练完成后执行以下命令：
```sh
tensorboard --logdir /tmp/retrain_logs
```

Once TensorBoard is running, navigate your web browser to `localhost:6006` to view the TensorBoard.
TensorBoard 运行之后，打开浏览器进到 `localhost:6006` 地址去看 TensorBoard 的状态。
The script will log TensorBoard summaries to `/tmp/retrain_logs` by default. You can change the directory with the `--summaries_dir` flag.
上面的脚本默认会将 TensorBoard 的运行日志记录到 `/tmp/retrain_logs`目录下。你也可以使用 `--summaries_dir` 标记来改变存储目录。
The [TensorBoard's GitHub](https://github.com/tensorflow/tensorboard) has a lot more information on TensorBoard usage, including tips & tricks, and debugging information.
【TensorBoard 的 GitHub 主页】(https://github.com/tensorflow/tensorboard) 有 TensorBoard 用法的更多信息，
包括一些小的提示，技巧和调试信息。
## Using the Retrained Model
## 再训练模型的使用
The script will write out a version of the Inception v3 network with a final
layer retrained to your categories to /tmp/output_graph.pb, and a text file
containing the labels to /tmp/output_labels.txt. 
下面的脚本将产生一个版本的 Inception v3 网络，其最后一层已经使用 `/tmp/output_graph.pb` 下的分类
以及 `/tmp/output_labels.txt` 下的标签训练完毕。
These are both in a format that
the @{$image_recognition$C++ and Python image classification examples}
can read in, so you can start using your new model immediately. Since you've
replaced the top layer, you will need to specify the new name in the script, for
example with the flag `--output_layer=final_result` if you're using label_image.
这些都是以图像识别 C++ 以及 Python 图像分类能够读取的格式存储，所以你可以立即使用这些新模型。
由于你替换了顶层节点，所以你需要在脚本中指定一个新名字，例如如果你正在使用 label_image 的话，
你可以使用 `--output_layer=final_result` 标志进行改变。
Here's an example of how to build and run the label_image example with your
retrained graphs:
下面的例子将展示如何使用你已经训练过的图来构建和运行 label_image 示例。
```sh
bazel build tensorflow/examples/image_retraining:label_image && \
bazel-bin/tensorflow/examples/image_retraining/label_image \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--output_layer=final_result:0 \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```

You should see a list of flower labels, in most cases with daisy on top
(though each retrained model may be slightly different). You can replace the
`--image` parameter with your own images to try those out, and use the C++ code
as a template to integrate with your own applications.
你将看到一组话的标签，大多数情况下以雏菊开头（尽管每个在训练模型可能有稍有区别）。你可以将
`--image` 的参数指定为你自己的图片来把之前的替换掉，并使用 C++ 代码作为模板来创建你自己的应用。

If you'd like to use the retrained model in your own Python program, then the
above
[`label_image` script](https://www.tensorflow.org/code/tensorflow/examples/image_retraining/label_image.py)
is a reasonable starting point.
如果你要在自己的 Python 程序中使用再训练模型，上面的[`label_image` 脚本](https://www.tensorflow.org/code/tensorflow/examples/image_retraining/label_image.py)
会是一个很好的开始模板。
If you find the default Inception v3 model is too large or slow for your
application, take a look at the [Other Model Architectures section](/tutorials/image_retraining#other_model_architectures)
below for options to speed up and slim down your network.
如果你觉得标准的 Inception v3 模型太大或者会使你你的程序变慢，你可以在[其他的模型结构](/tutorials/image_retraining#other_model_architectures)
寻找其他可以提升速度或者瘦身的方案。

## Training on Your Own Categories
## 在你自己的分类上进行训练
If you've managed to get the script working on the flower example images, you
can start looking at teaching it to recognize categories you care about instead.
如果你能够成功运行分类实例花朵图片的代码，你可以教他识别你关心的新分类。
In theory all you'll need to do is point it at a set of sub-folders, each named
after one of your categories and containing only images from that category. If
you do that and pass the root folder of the subdirectories as the argument to
`--image_dir`, the script should train just like it did for the flowers.
理论上所有你需要做的只是将训练对象指向一组新的子文件夹，每一个都你要训练的新分类命名，而且只包含符合这个分类的图片。
完成之后将这些子目录的上层根目录作为参数传给 `--image_dir`，脚本会像之前训练识别花朵一样完成训练过程。
Here's what the folder structure of the flowers archive looks like, to give you
and example of the kind of layout the script is looking for:
为了说明脚本搜寻的文件目录结构是怎样的，下图是花朵文件夹的目录结构：
![Folder Structure](https://www.tensorflow.org/images/folder_structure.png)
![目录结构](https://www.tensorflow.org/images/folder_structure.png)

In practice it may take some work to get the accuracy you want. I'll try to
guide you through some of the common problems you might encounter below.
实际操作中为了得到想要的准确性可能要做很多工作。我将通过下面一些你可能会遇到的常见问题进行讲解。
## Creating a Set of Training Images
## 创建一个用于训练的图片集
The first place to start is by looking at the images you've gathered, since the
most common issues we see with training come from the data that's being fed in.
我们要注意的第一个地方就是你所搜集的图片，我们发现训练过程中最容易出问题的是你输入的数据。
For training to work well, you should gather at least a hundred photos of each
kind of object you want to recognize. The more you can gather, the better the
accuracy of your trained model is likely to be. You also need to make sure that
the photos are a good representation of what your application will actually
encounter. For example, if you take all your photos indoors against a blank wall
and your users are trying to recognize objects outdoors, you probably won't see
good results when you deploy.
为了能使训练工作正常进行，你需要为你要训练识别的每个分类至少准备 100 张图片。你搜集的图片越多，
你训练出的模型越可能拥有更好的准确性。你同样需要确保你搜集的图片是你的应用将要识别的任务的很好的代表。
例如，你使用的图片都是背景是空樯的室内照片，而你的用户尝试去识别室外的物体，
当应用部署时你不会看到有好的效果
Another pitfall to avoid is that the learning process will pick up on anything
that the labeled images have in common with each other, and if you're not
careful that might be something that's not useful. 
另一个要避免的陷阱是学习过程会学习标签图片之间任何的相同之处，留意不要让它成为阻碍你的地方。
For example if you photograph one kind of object in a blue room, and another in a green one, then the model
will end up basing its prediction on the background color, not the features of
the object you actually care about. 
例如，如果你在蓝色的房间里给一个物体拍照，在另一个绿色的房间里给另一个物体拍照，那么最终模型将根据
背景的颜色给出预测，而不是依据你真正关心的物体特征。
To avoid this, try to take pictures in as
wide a variety of situations as you can, at different times, and with different
devices. If you want to know more about this problem, you can read about the
classic (and possibly apocryphal)
[tank recognition problem](https://www.jefftk.com/p/detecting-tanks).
为了避开这个陷阱，你要尽可能的在各种不同的环境中进行拍摄照片，不同的时间，使用不同的设备。
如果你想了解更多关于此问题的信息，你可以看一下经典的（也可能是杜撰的）
[坦克识别问题](https://www.jefftk.com/p/detecting-tanks).

You may also want to think about the categories you use. It might be worth
splitting big categories that cover a lot of different physical forms into
smaller ones that are more visually distinct. 
你同样需要考虑你要要使用的分类。应该将涵盖很多不同不同物理形体的大分类分割成在视觉上不同的小分类。
For example instead of 'vehicle' you might use 'car', 'motorbike', and 'truck'.
例如，应该使用 `car`, `motorbike` 和 `truck` 来代替 `vehicle`
It's also worth thinking about whether you have a 'closed world' or an 'open world' problem.
你同样应该思考你要解决的是一个封闭性问题还是一个开放性问题。
In a closed world, the only things you'll ever be asked to categorize are the classes of object you
know about. This might apply to a plant recognition app where you know the user
is likely to be taking a picture of a flower, so all you have to do is decide
which species. By contrast a roaming robot might see all sorts of different
things through its camera as it wanders around the world.
在封闭性问题中，你面对的问题只是识别你已经知道的的物体类别。这可以应用在一个植物识别应用中，
用户将拍摄一朵花的图片，你所要做的只是判定它的品类。
而一个满世界乱逛的漫游机器人可能通过它的相机看到各种各样的东西。
In that case you'd want the classifier to report if it wasn't sure what it was seeing. This can be
hard to do well, but often if you collect a large number of typical 'background'
photos with no relevant objects in them, you can add them to an extra 'unknown'
class in your image folders.
在那种情况下你可能需要分类器报告它是否确定你正在观察的东西。这可能很难做好，通常当你收集到一大批
典型的除了背景之外没什么东西的图片，你可以把他们加到额外的名为 unknown 的文件夹中。

It's also worth checking to make sure that all of your images are labeled
correctly. Often user-generated tags are unreliable for our purposes, for
example using #daisy for pictures of a person named Daisy. If you go through
your images and weed out any mistakes it can do wonders for your overall
accuracy.
检查确认所有的图片都与其标签相符合也是值得的。用户生成的标签经常性的并不可靠，例如使用标签 daisy 打给了
一个名叫 daisy 的人。如果你检查了所有的图片并排除了其中的错误，你会发现这对你模型整体的准确性有惊人的提升。
## Training Steps
## 训练步骤
If you're happy with your images, you can take a look at improving your results
by altering the details of the learning process. The simplest one to try is
`--how_many_training_steps`. 
如果你对你的图片很满意，你可以看一下通过调整学习过程来改善你的结果。最简单的一个选项是
`--how_many_training_steps`
This defaults to 4,000, but if you increase it to
8,000 it will train for twice as long. The rate of improvement in the accuracy
slows the longer you train for, and at some point will stop altogether, but you
can experiment to see when you hit that limit for your model.
这个选项默认是 4000，但是你可以把它调高到 8000，这样它大概需要两倍的时间完成训练。训练时间越长，
准确性的提高速率就会越低，最终准确性会停在某个点上，不过你可以实验一下你的模型什么时候会达到这个限制。

## Distortions
## 扭曲失真
A common way of improving the results of image training is by deforming,
cropping, or brightening the training inputs in random ways.
改善图片训练结果的一个一般方法是以随机的方式变形、裁剪或调亮训练的输入。
This has the advantage of expanding the effective size of the training data thanks to all the
possible variations of the same images, and tends to help the network learn to
cope with all the distortions that will occur in real-life uses of the
classifier. 
由于同一张图片可以衍生出各种可能的变种，这可以扩展有效训练数据集的大小，而且这可以帮助网络学习
应对各种变形，这种变形在现实生活的使用中会经常碰到。
The biggest disadvantage of enabling these distortions in our script
is that the bottleneck caching is no longer useful, since input images are never
reused exactly. This means the training process takes a lot longer, so I
recommend trying this as a way of fine-tuning your model once you've got one
that you're reasonably happy with.
允许变形输入的最主要缺陷是代码中对 bottleneck 的缓存将不再有效，因为输入的图片不会再被重复使用。
这意味着训练过程将多花很多时间，因此我建议只是将这种方法作为你已经训练完成一个模型之后的一种对模型调优的方法。
You enable these distortions by passing `--random_crop`, `--random_scale` and
`--random_brightness` to the script. These are all percentage values that
control how much of each of the distortions is applied to each image. 
你可以通过传入 `--random_crop`，`--random_scale`，以及 `--random_brightness` 开启扭曲功能。
这些都是控制这些扭曲方法作用于每张图片程度的百分比值。
It's reasonable to start with values of 5 or 10 for each of them and then experiment
to see which of them help with your application. `--flip_left_right` will
randomly mirror half of the images horizontally, which makes sense as long as
those inversions are likely to happen in your application. For example it
wouldn't be a good idea if you were trying to recognize letters, since flipping
them destroys their meaning.
开始时给每个选项设置 5 或 10 是合理的，然后再实验具体哪些值会对你的应用有所帮助。
`--flip_left_right` 将随机地水平镜面反映一半的图像，只要这些镜面对称图像有可能出现在你的应用中，
你设置此选项就是合理的。例如镜面对称出现在文字识别中就不是一个好主意，因为文字的反转会破坏它们的语义。

## Hyper-parameters
## 高层参数
There are several other parameters you can try adjusting to see if they help
your results. The `--learning_rate` controls the magnitude of the updates to the
final layer during training. Intuitively if this is smaller then the learning
will take longer, but it can end up helping the overall precision. That's not
always the case though, so you need to experiment carefully to see what works
for your case. The `--train_batch_size` controls how many images are examined
during one training step, and because the learning rate is applied per batch
you'll need to reduce it if you have larger batches to get the same overall
effect.
你还可以调节另外几个参数，看它们是否会对你的结果有所帮助。`--learning_rate` 控制
训练过程中对最后一层进行更新的量级。直观的可以知道，如果这个值更小，那么学习过长将更长，
但是却对最终整体的精确性有所帮助。情况也并不总是如此，因此你需要仔细的实验确认哪种情况对你有效。
`--train_batch_size` 控制一次训练步骤有多少图片会被检测。由于学习速率应用于每一个批次，
所以想每批使用更多的图片而整体效果不变，你需要减小学习速率。

## Training, Validation, and Testing Sets
## 训练、验证和测试集
One of the things the script does under the hood when you point it at a folder
of images is divide them up into three different sets. The largest is usually
the training set, which are all the images fed into the network during training,
with the results used to update the model's weights.
当你把脚本指向一个图片文件夹时所做的其中一件事是把他们划分成三个不同的集合。通常最大的集合为训练集，
这里的全部图片都用与训练的输入，并用图片的结果来不断调整模型的权重。
You might wonder why we
don't use all the images for training? A big potential problem when we're doing
machine learning is that our model may just be memorizing irrelevant details of
the training images to come up with the right answers.
你可能会有疑问，为什么不把所有的图片都用来训练。当我们做机器学习时的一个很大的潜在问题是
模型可能会通过记住大量的不相关的细节来得出正确的答案。
For example, you could
imagine a network remembering a pattern in the background of each photo it was
shown, and using that to match labels with objects. It could produce good
results on all the images it's seen before during training, but then fail on new
images because it's not learned general characteristics of the objects, just
memorized unimportant details of the training images.
例如，你能想象一个网络记住了展示给它的每一张相片的背景的模式，并使用它在物体和标签之间做匹配。他可能
会在之前训练过程中见到过的图片上做出很好的预测，但是却会在新的图片上失败，因为它没有学会物体的通用特征，
只是记住了训练图片上的一些不重要的细节。
This problem is known as overfitting, and to avoid it we keep some of our data
out of the training process, so that the model can't memorize them. We then use
those images as a check to make sure that overfitting isn't occurring, since if
we see good accuracy on them it's a good sign the network isn't overfitting. The
usual split is to put 80% of the images into the main training set, keep 10%
aside to run as validation frequently during training, and then have a final 10%
that are used less often as a testing set to predict the real-world performance
of the classifier. These ratios can be controlled using the
`--testing_percentage` and `--validation_percentage` flags. In general
you should be able to leave these values at their defaults, since you won't
usually find any advantage to training to adjusting them.
这个问题就是人们常说的过拟合，要避免这个问题就要让一部分数据不参与训练过程，以防模型会记住它们。
我们可以把不参与训练过程的那些数据作为模型是否过拟合的检验，如果模型在这些数据上有很好的准确性，
说明模型没有过拟合。通常将全部数据的 80% 作为主要的训练集，另外 10% 作为训练过程中的验证也会经常使用，
剩下的 10% 作为预测分类器在真实世界中的表现的测试数据集，不经常使用。划分的比例可以通过 `--testing_percentage` 
和 `--validation_percentage` 进行控制。一般情况下使用默认值就可以了，通常情况下
你不会因为调整它们而获得任何训练的优势。
Note that the script uses the image filenames (rather than a completely random
function) to divide the images among the training, validation, and test sets.
This is done to ensure that images don't get moved between training and testing
sets on different runs, since that could be a problem if images that had been
used for training a model were subsequently used in a validation set.
注意脚本通过文件名（而不是随机函数）对训练、验证和测试集中的图片进行区分。这么做是为了确保图片不会
在不同的运行期内在训练和测试集之间移动，因为如果用于训练的图片在随后又被用于验证，这会是一个问题。
You might notice that the validation accuracy fluctuates among iterations. Much
of this fluctuation arises from the fact that a random subset of the validation
set is chosen for each validation accuracy measurement. The fluctuations can be
greatly reduced, at the cost of some increase in training time, by choosing
`--validation_batch_size=-1`, which uses the entire validation set for each
accuracy computation.
你可能注意到验证的准确性在迭代中有波动。大部分的波动源于这样一个事实，每一次验证的准确率是由一个随机的验证子集来衡量的。
这种波动能被极大的减小，付出的代价是更多的训练时间，通过指定 `--validation_batch_size=-1`
可以使每一次准确率计算都使用全部的验证集。
Once training is complete, you may find it insightful to examine misclassified
images in the test set. This can be done by adding the flag
`--print_misclassified_test_images`. This may help you get a feeling for which
types of images were most confusing for the model, and which categories were
most difficult to distinguish. For instance, you might discover that some
subtype of a particular category, or some unusual photo angle, is particularly
difficult to identify, which may encourage you to add more training images of
that subtype. Oftentimes, examining misclassified images can also point to
errors in the input data set, such as mislabeled, low-quality, or ambiguous
images. However, one should generally avoid point-fixing individual errors in
the test set, since they are likely to merely reflect more general problems in
the (much larger) training set.
训练一旦完成，你会发现检查测试集中被错误分类的图片是一件极具洞察的事。可以通过添加 `--print_misclassified_test_images` 
查看这些图片。这可以帮助你获得关于哪种类型的图片对你的模型最有迷惑性，哪种分类最难以辨别的感性认识。
例如，你可能发现一些特定类别的子分类或者某些刁钻的拍摄角度特别难以识别，这可能激发你添加更多的那种子类的训练图片。
通常，检查错误分类的图片同样能指出输入数据中的一些错误，如标签错误、图片质量低、或者模棱两可的图片。
然而，通常应该避免对测试集进行针对性修复单个错误，因为这些错误很可能只是对训练集（数据量更大）中存在的更一般问题的一种反映。
## Other Model Architectures
## 其他模型结构
By default the script uses a pretrained version of the Inception v3 model
architecture. This is a good place to start because it provides high accuracy
results, but if you intend to deploy your model on mobile devices or other
resource-constrained environments you may want to trade off a little accuracy
for much smaller file sizes or faster speeds. To help with that, the
[retrain.py script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py)
supports 32 different variations on the [Mobilenet architecture](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html).
脚本默认使用 Inception v3 模型架构的预训练版本。这是一个很好的开始因为它提供的结果有很高的精确性，但是如果
你要把你的模型部署到移动设备或者资源有限的环境中时，你可能需要牺牲一点准确性以获得更小的文件体积或者更快的速度。
为了能帮助做到这些，这个[retrain.py script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py)
支持 32 中不同的[移动架构](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)衍生版本
These are a little less precise than Inception v3, but can result in far
smaller file sizes (down to less than a megabyte) and can be many times faster
to run. To train with one of these models, pass in the `--architecture` flag,
for example:

相比 Inception v3 这些衍生版本精确度要差一些，但是文件体积也小得多（小于 1 M 字节）而且运行速度要快上许多倍。
要使用这些模型进行训练，使用 `--architecture` 标记，例如：

```
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos --architecture mobilenet_0.25_128_quantized
```

This will create a 941KB model file in `/tmp/output_graph.pb`, with 25% of the
parameters of the full Mobilenet, taking 128x128 sized input images, and with
its weights quantized down to eight bits on disk. You can choose '1.0', '0.75',
'0.50', or '0.25' to control the number of weight parameters, and so the file
size (and to some extent the speed), '224', '192', '160', or '128' for the input
image size, with smaller sizes giving faster speeds, and an optional
'_quantized' at the end to indicate whether the file should contain 8-bit or
32-bit float weights.

它会在 `/tmp/output_graph.pb` 下创建一个 941KB 大小的模型文件，拥有完整 Mobilenet 参数的
25%，接受大小为 128x128 的图像文件作为输入，而且使用 8 个比特位表示权重。你可以使用 ‘1.0’，‘0.75’，
‘0.50’ 或者 ‘0.25’ 来控制权重参数的数量，还可以使用 ‘224’，‘192’，‘160’ 或者 ‘128’来
控制输入图像文件的大小（某种程度上也是在控制速度），输入的体积越小训练速度也就越快，最后还有一个可选的
`_quantized` 表示文件是否应该包含 8 位或 32 位浮点权重值。

The speed and size advantages come at a loss to accuracy of course, but for many
purposes this isn't critical. They can also be somewhat offset with improved
training data. For example, training with distortions allows me to get above 80%
accuracy on the flower data set even with the 0.25/128/quantized graph above.
速度和体积的优势当然会带来损失准确性，但是对于许多应用来说这并不是最关键的。损失的准确性可以通过
改善训练数据得到一些补偿。例如，使用变形图片让我在即使使用 0.25/128/quantized 的
输入图片配置时依然可以得到超过 80% 的准确性。
If you're going to be using the Mobilenet models in label_image or your own
programs, you'll need to feed in an image of the specified size converted to a
float range into the 'input' tensor. Typically 24-bit images are in the range
[0,255], and you must convert them to the [-1,1] float range expected by the
model with the formula  `(image - 128.)/128.`.
如果你要在标签图片或者你自己的程序中使用 Mobilenet 模型，你需要输入被转换成一个浮动的区间的特定大小的图片到
`input` 张量中。典型的 24 位图片的范围是 [0, 255]，你必须把他们通过公式 `(image - 128.)/128.` 转换到模型期望的 [-1, 1] 浮动区间内。

