import torch
import numpy as np
'''
# 创建一个3x3全1矩阵，Tensor
x1 = torch.ones(3, 3)
print(x1)

# 5x5 全0矩阵
x2 = torch.zeros(5, 5)
print(x2)

# 与x1同维0矩阵
x4 = torch.zeros_like(x1)
print(x4)

# 与x1同维全1矩阵
x5 = torch.ones_like(x1)
print(x5)

# 对角矩阵
x6 = torch.diag(torch.from_numpy(np.array([1, 2, 3, 4, 5])))
print(x6)

# 5x5 随机矩阵
x7 = torch.rand(5, 5)
print(x7)

# 5x5 norm分布矩阵
x8 = torch.randn(5, 5)
print(x8)

# 创建一个empty Tensor
x9 = torch.empty(3, 3)
print(x9)

# 创建一个Tensor,给定数值
x10 = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(x10)

# 根据已有的Tensor创建一个新的Tensor
x11 = torch.rand_like(x10, dtype=torch.float)
print(x11)

# 获取Tensor的size,  Tensor.Size 实际上是一个Tuple
print(x11.size())

# Tensor的in place 操作,会改变Tensor本身
x12 = torch.rand(3, 3)
print(x12.t_())
print(x12.copy_(x11))

# Tensor  resize/reshape操作
x13 = torch.randn(4, 4)
x14 = x13.view(16)          #16*1
print(x14)
x15 = x13.view(-1, 8)       # -1, 此维度根据另一个维度计算得到
print(x15)

# 只有一个数的Tensor,  使用xxx.item() 得到python 数值
x16 = torch.rand(1)
print(x16)
print(x16.item())

# 获取Tensor中元素的个数
x17 = torch.randn(1, 2, 3, 4, 5)
print(torch.numel(x17))
print(torch.numel(torch.zeros(4, 5)))

# 判断一个对象是否是Tensor
print(torch.is_tensor(x17))
print(torch.is_tensor(np.array([1, 2, 3, 4])))

# 判断一个对象是否为Pytorch storage object
print(torch.is_storage(torch.empty(3, 3)))          #False???
print(torch.is_storage(np.zeros(shape=(3, 3))))     #False

# 设置Tensor的数据类型，初始默认为torch.float32
print(torch.Tensor([1.2, 3]).dtype)                 #float32
torch.set_default_dtype(torch.float64)
print(torch.Tensor([1.2, 3]).dtype)                 #float64

# 获取默认的Tensor数据类型
print(torch.get_default_dtype())                    #float64
torch.set_default_dtype(torch.float32)
print(torch.get_default_dtype())                    #float32
'''


'''
pytorch矩阵操作
var = torch.Tensor()  返回一个Tensor

tensor1 = torch.Tensor(3, 3)
tensor2 = torch.Tensor(3, 3)
 var2 = torch.add(tensor1, tensor2)     # 矩阵加
 var3 = torch.sub(tensor1, tensor2)     # 减
 var4 = torch.mul(tensor1, tensor2)     # 乘
 var5 = torch.div(tensor1, tensor2)     # 矩阵点除
 var6 = torch.mm(tensor1, tensor2)      # 矩阵乘

'''

x1 = torch.Tensor(5, 3)  # 构造一个5x3的Tensor
x2 = torch.rand(5, 3)  # 构造一个随机初始化的Tendor
print(x1.size())
print(x2.size())

# #####################Tensor相加################################
# 2个Tensor相加
y = torch.rand(5, 3)
var1 = torch.add(x1, y)

# 2个Tensor相加
var2 = x2 + y

var3 = torch.rand(5, 3)
# 2个Tensor相加
torch.add(x1, y, out=var3)
print(var3)

# Tensor相加，y.add_() 会改变y的值
y.add_(x2)
print(y)

# #####################Tensor相减###############################
x3 = torch.rand(5, 5)
x4 = torch.rand(5, 5)

y2 = x3 - x4
y3 = torch.sub(x3, x4)
print(x3.sub_(x4))

# ###################Tensor相乘################################
x5 = torch.rand(3, 3)
x6 = torch.rand(3, 3)
y4 = x5 * x6  # 矩阵元素点乘
y5 = torch.mul(x5, x6)
print(x5.mul_(x6))

# ###################Tensor相除################################
x7 = torch.rand(5, 5)
x8 = torch.rand(5, 5)
y6 = x7 / x8
y7 = torch.div(x7, x8)
print(x7.div_(x8))

# #################Tensor矩阵乘################################
x9 = torch.rand(3, 4)
x10 = torch.rand(4, 3)
y8 = torch.mm(x9, x10)
# print(x9.mm_(x10))    # 错误用法


# ###################矩阵切片###################################
# 矩阵切片
var4 = x2[:, 1]
print(var4)