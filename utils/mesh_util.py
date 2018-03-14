'''
mesh utils
Tianye Li
'''

import numpy as np

# -----------------------------------------------------------------------------

class Mesh():
    def __init__( self, v=None, f=None, vc=None, vn=None ):
        self.v  = v
        self.f  = f
        self.vc = vc # currently need manually specify
        self.vn = vn # currently need manually specify
    def write_obj( self, filename ):
        save_obj( self, filename )
    def copy( self ):
        return Mesh( v=self.v, f=self.f, vc=self.vc, vn=self.vn )
    def initialize_vc( self ):
        self.vc = np.ones_like( self.v )

# -----------------------------------------------------------------------------

def load_obj( filename ):
    # based on Shunsuke Saito's code
    # todo: support loading per-vertex color
    f = open(filename, 'r')
    V = []
    F = []
    VC = []
    VN = []
    for line in f:
        line = line.rstrip('\n')
        parts = line.split(" ")
        if parts[0] == 'v':
            parts.remove('v')
            v = [float(a) for a in parts]
            if len(v) == 6:
                VC.append(v[3:6])
            V.append(v[0:3])
        elif parts[0] == 'f':
            parts.remove('f')
            face = [int(float(ft.split('//')[0]))-1 for ft in parts] # TODO: problematic if used for rendering (all 3 vertices per triangle needed)
            F.append(face)
        if parts[0] == 'vn':
            parts.remove('vn')
            vn = [float(a) for a in parts]
            VN.append(vn[0:3])
    f.close()

    if len(VC) == 0:
        mesh_vc = None
    else:
        mesh_vc = np.asarray( VC ).reshape((-1,3))

    if len(VN) == 0:
        mesh_vn = None
    else:
        mesh_vn = np.asarray( VN ).reshape((-1,3))

    return Mesh( v=np.asarray(V).reshape((-1,3)), 
                 f=np.asarray(F).reshape((-1,3)),
                 vc=mesh_vc,
                 vn=mesh_vn )

# -----------------------------------------------------------------------------

def save_obj( mesh, filename ):
    # based on Shunsuke Saito's code
    # support per-vertex color and normals
    # https://en.wikipedia.org/wiki/Wavefront_.obj_file
    
    V = mesh.v.ravel()
    F = mesh.f
    file = open(filename, "w")

    # write v and vc
    if mesh.vc is None:
        for i in range(V.shape[0]//3):
            file.write('v %f %f %f\n' % ( V[3*i], V[3*i+1], V[3*i+2] ) )
    else:
        VC = mesh.vc.ravel()
        for i in range(V.shape[0]//3):
            file.write('v %f %f %f %f %f %f\n' % ( V[3*i], V[3*i+1], V[3*i+2], VC[3*i], VC[3*i+1], VC[3*i+2] ) )
    
    # write vn and f
    if mesh.vn is not None:
        VN = mesh.vn
        for i in range(VN.shape[0]):
            file.write('vn %f %f %f\n' % (VN[i,0], VN[i,1], VN[i,2]))

        # write f for vertices and normals
        if F is not None:
            for i in range(F.shape[0]):
                file.write('f %d//%d %d//%d %d//%d\n' % (F[i,0]+1, F[i,0]+1, F[i,1]+1, F[i,1]+1, F[i,2]+1, F[i,2]+1))

    else:
        # write f
        if F is not None:
            for i in range(F.shape[0]):
                file.write('f %d %d %d\n' % (F[i,0]+1, F[i,1]+1, F[i,2]+1))
    
    file.close()
