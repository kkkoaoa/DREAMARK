import numpy as np

cdef extern from "pyarray_symbol.h":
    pass

cimport numpy as np

np.import_array()

cdef extern from "pywrapper.h":
    cdef object c_dual_contouring "dual_contouring"(np.ndarray, np.ndarray) except +

def dual_contouring(np.ndarray dist, np.ndarray normal):
    verts, faces = c_dual_contouring(dist, normal)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces
