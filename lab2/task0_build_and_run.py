from Viewer.controller import Controller, SimpleViewer

def main():
    viewer = SimpleViewer()
    # 创建一个控制器，是输入信号的一层包装与平滑
    # 可以使用键盘(方向键或wasd)或鼠标控制视角
    # 对xbox类手柄的支持在windows10下测试过，左手柄控制移动，右手柄控制视角
    # 其余手柄(如ps手柄)不能保证能够正常工作
    # 注意检测到手柄后，键盘输入将被忽略
    controller = Controller(viewer)
    viewer.run()

if __name__ == '__main__':
    main()