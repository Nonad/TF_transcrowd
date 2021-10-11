# TF_TransCrowd
Learning methods of counting tasks

Learning methods depended on Transformer

Trying to refactor VisionTransformer from timm and Transcrowd depened on Pytorch  with TensorFlow

https://github.com/rwightman/pytorch-image-models

&

https://github.com/dk-liang/TransCrowd

Having little knowledge about license and pt and tf

Tell me if there was any wrong usage plz and thank you more than a thousand times

owo





> （1）keras更常见的操作是通过继承Layer类来实现自定义层，不推荐去继承Model类定义模型，详细原因可以参见官方文档
>
> （2）pytorch中其实一般没有特别明显的Layer和Module的区别，不管是自定义层、自定义块、自定义模型，都是通过继承Module类完成的，这一点很重要。其实Sequential类也是继承自Module类的。
>
> 注意：我们当然也可以直接通过继承torch.autograd.Function类来自定义一个层，但是这很不推荐，不提倡，至于为什么后面会介绍。
> ————————————————
> 版权声明：本文为CSDN博主「LoveMIss-Y」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/qq_27825451/article/details/90550890
