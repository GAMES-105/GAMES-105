from Viewer.controller import Controller, SimpleViewer

def main():
    viewer = SimpleViewer(float_base = True, substep = 32)
    viewer.run()

if __name__ == '__main__':
    main()