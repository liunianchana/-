#导入图像模块
import Module_image as Mod
from PIL import Image

input_file='D:/python学习/untitled/数据/images'


# 读取所有图片的名称
print("开始读取所有图片的名称")
imgs_list = Mod.RIL(input_file)
print("读取所有图片的名称已完成")

# 初始化图片特征数据
print("初始化图片特征数据")
data = []

# 图片转换
print("开始进行图片处理")
for i in range(len(imgs_list)):

    # 读取图片
    img = Image.open(input_file+'/'+imgs_list[i])

    # 进行图片分割
    div = Mod.Division(img)

    # 图片特征提取
    Feature = Mod.Features(div)

    data.append(Feature)
    print("正在进行图片处理"+str(i))

print("已完成图片处理")

# 数据标准化
print("开始数据标准化")
data_standard = Mod.Standard(data)
print("已完成数据标准化")

# 对数据进行变形
print("开始对数据进行变形")
data = Mod.Changes(data)
print("已完成数据变形")

# 添加类别列与序号列
print("开始添加类别列与序号列")
data = Mod.Category(imgs_list,data)
data = Mod.Order(imgs_list,data)
data_standard = Mod.Category(imgs_list,data_standard)
data_standard = Mod.Order(imgs_list,data_standard)
print("已完成添加类别列与序号列")

# 保存数据
print("保存数据")
data.to_csv('D:/result/data.csv', encoding='gbk', index=False)
data_standard.to_csv('D:/result/data_standard.csv', encoding='gbk', index=False)
print("程序执行完毕")