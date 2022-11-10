# 作业1

## 任务零  环境配置

### 1. 安装panda3d(Windows/Linux/Mac)
   
建议使用conda等建立新的虚拟环境
```bash
conda create -n games105 python=3.8
conda activate games105
conda install numpy scipy
pip install panda3d
```
如果下载过慢可使用清华镜像源安装 ( https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/ )

> 本作业只允许使用 `numpy`， `scipy`，`pytorch`(`tensor-flow`, `jitter`)以及其依赖的库。评测时也以此为准。版本限定到2022年10月31日之前的发布版本。作业文件中请不要import除此之外的库。

完成后可以运行`task0_test_environment.py`，你将会看到一个T-pose的人形角色。可以通过鼠标左右中键操作相机视角。

**注意：** `scipy`的`from_euler`里`xyz`大小写代表不同的旋转方式，本次作业里你需要一律使用大写的`XYZ`。

### 2. 一些角色动画领域的术语和Scipy-Rotation库简介(可跳过)

* Body/Link: 角色的身体的某一部分，是一个空间中实际呈现的物体。比如说，UpperLeg表示角色膝盖往上的大腿部分。
* Joint: 关节，是连接两个Body的枢纽。作为约束使得两个Body不分开。比如说，Elbow连接角色的大臂和小臂，叫做肘关节。
* 树形结构: 由于角色是每两个相邻的Body之间被Joint约束，每两个Joint之间又会夹着一个Body。可以发现角色有着树形的拓扑结构。于是可以定义Parent-Body、Parent-Joint。
* RootBody: 在角色动画领域，常常把Pelvis(骨盆)作为最原始的Body，其他的一些Body如LowerBack(脊椎下部)、LeftUpperLeg、RightUpperLeg作为其Child-Body。
* RootJoint: 为了方便常常在RootBody的中心位置虚拟一个Joint，作为最原始的Joint。这样的结果是使得Joint数目和Body的数目相等。
* Pose: 姿态。由于Body和Joint互相约束，所以描述角色“姿态如何”可以用Body也可以用Joint。通常，用表示空间取向(Orientation)和空间位置(Position)来描述。也就是Pose = Orientation $\cup$ Position。
* BVH文件: 一种常见的描述角色Pose的数据格式。由于BVH文件不含有Body的信息，里面的信息只会描述Joint的Pose。详细的介绍BVH结构及旋转的网页请参考 https://zhuanlan.zhihu.com/p/539522515
* Local/Global: 实际上，由于角色有着树形结构，采用一个Joint在其Parent-Joint的坐标系的相对旋转也可以描述空间取向。可见，Orientation是Global表述，其对应的Local表述叫Rotation，特指在Parent-Joint坐标系下的相对旋转。Position也是Global表述，对应的Local表述叫Translation或者Offset。不难推导，如果我们知道从RootJoint到某一个子孙Joint（记为$Joint_{i}$）的一条路径上所有的Rotation。那么$Orientation_i=Rotation_0\cdot Rotation_1 ... \cdot Rotation_i$。(请自行推导Position和Translation的关系:-)
* Scipy的Rotaion库
```python
from scipy.spatial.transform import Rotation as R
```
四元数、旋转矢量、欧拉角、旋转矩阵从Python-List或者Numpy-Ndarray读入
```python
quat = R.from_quat([ 0.        ,  0.80976237,  0.53984158, -0.22990426])
rot_vec = R.from_rotvec([0, 3, 2])
euler_angle = R.from_euler('XYZ', [-109.60643629,  -21.85973481, -164.48716608], degrees=True)
matrix = R.from_matrix([[-0.89428806,  0.24822376, -0.37233564],
       [-0.24822376,  0.41714214,  0.8742868 ],
       [ 0.37233564,  0.8742868 , -0.31143019]])
```
在不引起歧义的时候，可以使用`R(·)`读入
```python
quat = R([ 0.        ,  0.80976237,  0.53984158, -0.22990426])
```
打印输出/互相转化
```python
quat = R.from_rotvec([0, 3, 2]).as_quat()
euler_angle = R.from_matrix([[-0.89428806,  0.24822376, -0.37233564],
       [-0.24822376,  0.41714214,  0.8742868 ],
       [ 0.37233564,  0.8742868 , -0.31143019]]).as_euler('XYZ', degrees=True)
matrix = R.from_quat([ 0.        ,  0.80976237,  0.53984158, -0.22990426]).as_matrix()
```
其他较为细节的参数指定和复杂用法请阅读https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

如果通过`R.from_rotvec()`,`R.from_quat()`,`R.from_euler()`,`R.from_rotvec()`来将“广义旋转”读入后，可以用`*`进行旋转算符操作，有以下关系：
```python
a = R.from_rotvec([0, 3, 2])
b = R.from_rotvec([2, 1, 5])
c = R.from_rotvec([4, 7, 3])
ans1 = (a * b * c).as_matrix()
ans2 = a.as_matrix() @ b.as_matrix() @ c.as_matrix()
```
在误差允许范围内，`ans1==ans2`



## 任务一 前向运动学
在这部分中，你需要实现前向运动学函数。

本部分的运行代码在`task1_forward_kinematics.py`，你需要完成的代码在`Lab1_FK_answers.py`。测评时我们将单独拿出`Lab1_FK_answers.py`，你可以对其进行任何必要修改，但请保持接口不变。

在`task1_forward_kinematics.py`的main()函数中，我们定义了一系列测试任务，你需要根据顺序完成需要的函数(在`Lab1_FK_answers.py`中完成)。

在每个部分你需要取消注释对应的测试代码，如
```
part1(viewer, bvh_file_path)
```

### Part 1: 读入BVH文件的人体结构

在这一部分，你需要完成`part1_calculate_T_pose`函数。它输入一个bvh文件路径，返回包括关节的名字, 父关节的索引, 所有关节的偏移量。

提示: 由于虚拟角色是一个树形结构, 可以通过递归的方式实现, 也可以手动模拟一个栈(Stack)。

> 注意需要包含根节点和末端，根节点起名为RootJoint，末端起名为父节点+'_end'

> 为了方便起见，假设BVH文件的Channels无论Translation还是Rotation均按XYZ顺序排列，在用scipy读取时，可以用`R.from_euler('XYZ', [-109.60643629,  -21.85973481, -164.48716608], degrees=True)`类似的格式

> 默认Channel设置是RootJoint节点有6个channel(平移和旋转)，其余子节点有3个，末端节点没有channel。可以参考`data/walk60.bvh`

在实现后，请在`main`函数内取消part1部分的注释，以显示part1的结果，正确读入后, 人会保持T-pose站立(由于腰部在原点，所以会有下半身在地下，可以转动相机看到)。

### Part 2:前向运动学计算

在这一部分，你需要完成`part2_one_pose`函数。读入BVH文件中每一帧的动作数据, 计算每个关节在世界坐标系下的旋转和位置。

> 虚拟角色是一个树形结构, 应该从Root结点开始, 向下遍历计算每个关节的旋转和位置。这里需要注意, BVH上面的channel和下面每一行的数据是对应的。我们需要考虑每个关节的欧拉角顺序。以及, 注意在BVH文件里, 是角度[-180, 180], 不是弧度。 

`load_motion_data`辅助函数会为你读出bvh内所写的动作内容。这部分里我们为你调用并将返回的数据当作了函数的输入，之后你也可能会用到它。

在`task1_forward_kinematics.py`，需要写出在使用给出的`load_motion_data`辅助函数读出的Local的Joint Translation和Rotation数据之后，计算Global的Joint Position和Orientation。请在`task1_forward_kinematics.py`中的`part2_forward_kinematics`函数内完成这一任务。
> 需要注意的是我们提供的设置Joint四元数的顺序为(x, y, z, w)，请符合这一顺序，否则渲染结果错误

当任务完成后，请在`main`函数内取消part2部分的注释，其中`part2_one_pose`为选取某一帧来显示的角色姿态，`part2_animation`为完整播放BVH动画。

### Part 3:运动重定向

读入一个A-Pose的文件, 将A-pose的bvh重定向到T-pose上
> A-Pose是指角色的初始姿态为肩膀下垂。和T-Pose的差别是肩膀的局部旋转不同。如下图：

<div align="center">
 <img src="images/apose.png" width = "45%" >
</div>

我们希望你将其转变成一个以T-pose为初始姿态的bvh。故而我们调用你在part1所写的函数读取了一个T-pose的表示。希望你完成`part3_retarget_func`函数，返回一个`motion_data`。我们会将T-pose和你返回的`motion_data`输入到你在part2所写的函数中，希望获取正确的运动结果。

> 我们这里采用的A-Pose是在Local-Rotation上将lshoulder用欧拉角XYZ的格式旋转[0, 0, -45]度，将rshoulder用欧拉角XYZ的格式旋转[0, 0, 45]度。虽然有了两个bvh的你应该不需要这个信息，但是它会帮你极大简化工作量。

当任务完成后，请在`main`函数内取消part3部分的注释，会得到重定向后的运动动画。

### Part 4(选做):任意BVH重定向
在网络上有很多开源的BVH数据集，是由真人动作捕捉数据得到的。如LaFAN、SFU-Mocap数据集。

```
LaFAN: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
SFU-Mocap: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
```

它们的节点和我们的有可能不一致，如果有兴趣还是可以将其进行重定向来进行播放使用的。

## 任务二 逆向运动学

在这部分中，你需要实现一个简化的逆向运动学函数。你可以选择任何你喜欢的IK实现方式，比如CCD,雅各比等。各个任务中，IK精度要求都为误差距离`0.01`以内即可(如果能到达目标位置的话)。为了保证帧率，建议限制最大迭代次数。

本部分的运行代码在`task2_inverse_kinematics.py`，你需要完成的代码在`Lab2_IK_answers.py`。测评时我们将单独拿出`Lab2_IK_answers.py`，你可以对其进行任何必要修改，但请保持接口不变。

### Part 1: 静态逆向运动学

假设角色处于静止状态，起始为T-pose，你需要在此基础上调整姿态，保持根节点固定在原地不动，让其左手够到给定位置。

> 此过程相当于在操作一个从固定点开始，到左手为止的机械臂。而你需要获取这些可操控关节的索引。不过我们已经为你实现了一个获取可操控路径的函数，请你仔细阅读`MetaData`类，里面记录了你需要的一些固定信息，并提供了一个获取路径的函数。


本部分需要你完成的函数为`part1_inverse_kinematics`，在这个函数中你需要计算能使左手到达给定位置的角色关节位置和旋转，具体的输入输出要求可以看代码中的注释。

为了测试你实现的逆向运动学算法，我们设计了三个小问题，难度逐渐提高。在`part1_simple`中，我们将根节点设在腰部，即保持腰部位置不动，让左手够到输入的指定位置。在`part1_hard`中，根节点为左脚，最终实现的效果应该与下图相似，角色通过倾斜身体和弯曲手臂使左手够到指定位置。

<div align="center">
 <img src="images/inverse_kinematics_before.png" width = "45%" >
 <img src="images/inverse_kinematics_after.png" width = "45%" >
</div>

如果正确完成了part1_inverse_kinematics，那么在`part1_animation`中不用做任何额外代码，就可以通过`wasd`控制marker的位置，来实时改变角色的姿态，以保持角色的左手始终与marker位置重合。

### Part 2: 逆向运动学+bvh

输入一个bvh，要求角色的其他身体部分在按照bvh动作的同时，左手在水平方向上对根节点保持静止，竖直方向上维持同一个绝对高度。

在这部分中，你需要实现`part2_inverse_kinematics`函数，该函数的输入中包括左手相对于角色根节点的x方向、z方向目标位置，以及y方向的目标绝对高度，你的实现需要计算出左肩以下的关节位置及旋转，使左手保持在目标位置，同时身体其他部分的动作要和bvh一致。

实现完这个函数后，请在`task2_inverse_kinematics.py`的`main`部分中取消调用`part2`函数的注释，运行代码后的效果应该与下图相似。

<div align="center">
 <img src="images/ik_p2_1.png" width = "45%" >
 <img src="images/ik_p2_2.png" width = "45%" >
</div>

### Part 3(选做)： 两个目标的逆向运动学

与Part 1类似，这次我们同时给出左手和右手的目标位置，你需要实现逆向运动学算法使两只手都尽量满足目标要求

本部分需要在`bonus_inverse_kinematics`实现逆向运动学算法，计算出的角色姿态需要同时满足两只手的目标位置约束。

实现完成后，请在`task2_inverse_kinematics.py`的`main`部分中取消调用`bonus`函数的注释，运行检查效果。两个目标位置可以分别通过`wasd`和方向键控制。

## 提交

作业提交通过[http://cn.ces-alpha.org/course/register/GAMES-105-Animation-2022/](http://cn.ces-alpha.org/course/register/GAMES-105-Animation-2022/)，邀请码`GAMES-FCA-2022`。

需要提交的文件是`Lab1_FK_answers.py`和`Lab2_IK_answers.py`。如果你有想要展示的东西，也可以额外提交一个zip(文件大小请不要超过20M)

如有问题或Bug，可以在issue，QQ群和我们的[课程讨论版](https://github.com/GAMES-105/GAMES-105/discussions)内进行讨论。
