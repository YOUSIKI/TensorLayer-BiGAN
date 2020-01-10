# BiGAN.TensorLayer
又一个在celebA数据集上的BiGAN尝试。

代码的大致思路和另一条分支差不多，额外做的是对encoder的测试，即依照论文的结论，encoder应该大体上是generator的逆；对image和G(E(image))所有的测试结果展示在pair里。

示例图片（第一幅为真，第二幅为假）

![](D:\python程序\大二\project\BiGAN.TensorLayer\pair\real_05.png)

![](D:\python程序\大二\project\BiGAN.TensorLayer\pair\reproduced_05.png)



BiGAN的可逆应该只是很弱的可逆，论文里面也提到，对自然景物图片的训练，可逆仅仅是体现出原图片的部分特征。这一点在人脸数据集上体现比较明显，BiGAN的效果不怎么样。