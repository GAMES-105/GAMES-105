from panda3d.core import *
import numpy as np
def draw_circle(nodepath, radius, color):
    from direct.showutil.Rope import Rope
    r = Rope()
    w = 0.5
    a = 0.866*radius
    b = 0.5*radius
    c = 1*radius
    points = [ 
              (-a, 0, b,1),
              (0,0,c,w),
              (a,0,b,1),
              (a,0,-b,w),
              (0,0,-c,1),
              (-a,0,-b,w),
              (-a,0,b,1),
              ]
    verts = [
        {
            'node': nodepath,
            'point': point,
            'color': color,
            'thickness': 10,
        }
        for point in points
    ]
    r.ropeNode.setUseVertexThickness(1)
    r.ropeNode.setUseVertexColor(1)
    r.setup(3, verts, knots=[0,0,0,1/3,1/3,2/3,2/3,1,1,1])
    r.reparentTo(nodepath)
    return r

def draw_circle_with_arrow(nodepath, radius, color, with_circle = True):
    if with_circle:
        draw_circle(nodepath, radius, color)
    from direct.showutil import BuildGeometry as BG
    node = nodepath.attach_new_node('arrow')
    _,_,geo = BG.addArrowGeom(node, 0.03,0.3, color = Vec4(*(i/2 for i in color)))
    node.setQuat(Quat(0,0,1,0)*Quat(0.707,0.707,0,0))
    node.setPos(0,0,0.15)
    node.wrtReparentTo(nodepath)
    return node

def draw_arrow(nodepath, width, length, color):
    from direct.showutil import BuildGeometry as BG
    node = nodepath.attach_new_node('arrow')
    _,_,geo = BG.addArrowGeom(node, width,length, color = Vec4(*(i for i in color)))
    node.setQuat(Quat(0,0,1,0)*Quat(0.707,0.707,0,0))
    node.setPos(0,0,length/2)
    node.wrtReparentTo(nodepath)
    return node


def pos_vel_to_beizer(position, velocity, dt):
    position = position.reshape(-1,3)
    velocity = velocity.reshape(-1,3)
    prev = position - velocity*dt
    post = position + velocity*dt
    points = np.concatenate([prev, position, post], axis = -1).reshape(-1,3)
    points = points[1:-1]
    return points

def draw_beizer(positions, velocity, dt, rope):
    points = pos_vel_to_beizer(positions, velocity, dt)
    points = [ {
        'node': None,
        'point': tuple(point),
        'color': (0,0,1,1),
        'thickness': 10,
                } for point in points]
    if rope is None:
        from direct.showutil.Rope import Rope
        rope = Rope()
        rope.ropeNode.setUseVertexThickness(1)
        rope.ropeNode.setUseVertexColor(1)
    rope.setup(3, points)
    rope.set_render_mode_thickness(100)
    return rope
    # rope.verts = points
    # rope.recompute()