from .viewer import SimpleViewer
from direct.actor.Actor import Actor
from panda3d.core import *
import numpy as np
# import simplepbr
# loadPrcFileData("", "load-file-type fbx p3assimp")

class MeshViewer(SimpleViewer):
    def __init__(self, *args, **kwargs):
        super(MeshViewer, self).__init__(*args, **kwargs)
    
    def load_character(self):
        self.model = Actor(r"material/test.bam")
        self.model.children[0].setQuat(Quat(1,0,0,0))
        self.model.reparentTo(self.render)
        self.model.ls()
        
        for c in self.model.findAllMatches("**/+GeomNode"):
            c.find_all_materials()[0].setShininess(1.0)
            c.find_all_materials()[0].setEmission((0.6,0.6,0.6,0.6))

        geomNodeCollection = self.model.findAllMatches('**/+GeomNode')
        geom = geomNodeCollection[0].node().modifyGeom(0)
        self.vertexData = geom.modifyVertexData()
        
        self.init_abs_position = self._get_raw_position()    
    
    def _get_raw_position(self):
        array = self.vertexData.modifyArray(0)
        view = memoryview(array)
        nparray = np.frombuffer(view, dtype=np.float32).copy().reshape(-1,8)
        # nparray = np.frombuffer(view, dtype=np.float32).reshape(-1,6)
        return nparray[:,0:3]
    
    def set_vertex_position(self, position):
        position = position.astype(np.float32)
        array = self.vertexData.modifyArray(0)
        view = memoryview(array)
        nparray = np.frombuffer(view, dtype=np.float32).copy().reshape(-1,8)
        nparray[:,0:3] = position
        array.modify_handle().copyDataFrom(nparray.data)
        
    def get_skinning_matrix(self):
        
        name_list = [ x.name for x in self.model.getJoints()]
        
        num_joint = len(name_list)
        
        array = self.vertexData.getArray(2)
        view = memoryview(array)
        nparray = np.frombuffer(view, dtype=np.float32).copy().reshape(-1,5)
        index = nparray[:,0].copy()
        index = np.frombuffer(index.data, dtype=np.int8).reshape(-1,4)
        
        skinning_matrix = np.zeros((nparray.shape[0], num_joint))
        np.put_along_axis(skinning_matrix, index, nparray[:,1:], axis=1)
        return skinning_matrix, name_list, index, nparray[:,1:]