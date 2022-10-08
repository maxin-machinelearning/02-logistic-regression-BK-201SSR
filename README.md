# 2.Logistic回归（分类）

## 注：

1.`lab`为手动实现`Logistic`二分类代码及结果（使用梯度下降算法，推导过程见下文）

2.`Office`和`scipy_opt_example`为老师代码

3.`work`为基本作业。(Readme的所有内容也会放在其中)

4.`wine.data`为`lab`所使用的数据集https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

## 1.作业内容

基本要求：

- 使用`sklearn`实现`logistic`回归
- 回归问题：拟合数据： (选取合适的区间)
- 分类问题：生成随机样本点，采用标准数据集

提高练习：

- 尝试使用`spicy.optimize`中的优化算法训练`logistic regression`并与`sklearn`比较
- 尝试手动实现梯度下降**【必修】**

## **2.实现梯度下降原理**

### 	1.线性模型

$$
假设超平面方程：w^Tx+b=0\\
(1) 在超平面上方的点满足：w^Tx+b=0>0\\
(2) 在超平面下方的点满足：w^Tx+b=0<0\\
$$

### 	2.Logistic函数

$$
\sigma(z) = \frac {1} {1+e^{-z}}\\
$$

​	

### 	3.Logistic模型

$$
假设函数为：h_{w}(x) = \sigma(g(x)) = \frac {1} {1+e^{-w^Tx}}\\
即z = g(x) = w^Tx
$$

### 	4.极大似然估计求损失函数

$$
h_w(x)的概率意义为：P_{(y=1|x)}=h_w(x);P_{(y=0|x)}=1-h_w(x)\\
因此将实例x_i，预测为y_i的概率为：
\color{red}P_{(y=y_i|x_i;w)}=h_w{(x_i)}^{y_i}(1-h_w{(x_i)}^{1-y_i})\\
极大似然估计求得损失函数：
J(w)=-\frac{1}{m}\sum^m_{i=1}y_iIn(h_w(x_i))+(1-y_i)In(1-h_w(x_i))\\
$$

### 	5.梯度下降更新公式

$$
对J(w)求梯度\nabla{J(w)}:
\nabla{J(w)}=\frac{1}{m}\sum^m_{i=1}(h_w(x_i)-y_i)x_i\\
假设学习率为\eta,则模型参数w的更新公式为：
w:=w-\eta\nabla{J(w)}
$$



## 3.实验结果

见`lab.ipynb`

## 4.实验总结

#### 1.小结：

（1）通过改进Logistic回归函数，解决数据溢出问题

（2）使用`from sklearn.metrics import accuracy_score` 计算预测准确率

（3）使用`np.genfromtxt`读取data文件,获取数据集

#### 2.问题：

1、`RuntimeWarning ': divide by zero encountered in log`：运行时警告 '：在日志中遇到除以0

2、`overflow encountered in exp`：在 exp 中遇到溢出