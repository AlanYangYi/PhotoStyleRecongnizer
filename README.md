# PhotoStyleRecongnizer
Recognize the stlyle of photo taken by differnt sexual orientation's people  by using CNN

1 Use vgg19 feature layer and classification layer, freezing the parameters of vgg19 feature layer, and train classification layer parameters (transfer learning)

2 The optimizer chooses SGD. And at first tried to use Adam , but found that it is not suitable to freeze most layers of transfer learning and it shows that  the training set will converge difficultly.

3 Use the Dropout layer to prevent overfitting

4 In this experiment, the 98% accuracy rate identification of the training data was realized, and the 80% accuracy rate identification of the verification data was realized.

5 The average recognition accuracy rate of different test sets is about 60%. (The human recognition rate of the test set is about 45%. The test set address: link: https://pan.baidu.com/s/1oxp233_a0bXsWOhhrovY_g Extraction code: at0h)

6 The low accuracy rate may be due to the fact that the photos of straight men and gays are not very recognizable overall, which means that the distribution of differences between the photos of straight men and gays on social app will not be too different.
That means for most of the photos, it is difficult to accurately judge by the photos.
7 Later, the thermal activation map Grad-Cam was used to identify the activation area of the network
reference: https://arxiv.org/abs/1610.02391


8 Download URL of trained model:
Link: https://pan.baidu.com/s/1hepPgcbTX7vTvVHJmlGiiA Extraction code: uzwh (requires python environment or conda prompt)

* Photo data source: crawled from social apps



1 使用vgg19特征层+分类层，冻结vgg19特征层的参数，训练分类层参数（迁移学习）

2 优化器选用SGD，一开始尝试Adam，但发现不适合冻结大部分层的迁移学习，会出现learningRate过大情况，训练集无法收敛

3 使用Dropout层防止过拟合

4 实验中实现了训练数据的98%的准确率辨识，验证数据80%的准确率辨识

5 不同测试集辨识准确率平均水平大致为60%左右,（该测试集人类的辨识率为45%左右，测试集地址：链接:https://pan.baidu.com/s/1oxp233_a0bXsWOhhrovY_g 提取码:at0h ）

6 准确率低可能是因为直男和基友的照片差异整体上来说是没有很大的辨识度的，也就是说直男和基友在社交软件上的照片的差异分布不会相差太大
绝大部分照片，很难准确的通过照片去判断，只有一小部分可以

7 后期使用了类热激活图Grad-Cam来辨识网络的激活区域
https://arxiv.org/abs/1610.02391


8训练编译好的模型下载地址：
链接:https://pan.baidu.com/s/1hepPgcbTX7vTvVHJmlGiiA 提取码:uzwh  （需要python环境或conda prompt）

* 照片数据来源：从社交app上爬取
