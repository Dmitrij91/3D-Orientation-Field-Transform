' DIPY package required '


from dipy.sims.voxel import multi_tensor_odf
from dipy.data import get_sphere
from dipy.viz import window, actor



" Plot maximal response direction per voxel, Euler Angles must be in deg format "

def polar_from_cartesian(cor_car):
    """ Embedded 3D unit vector from spherical polar coordinates.
    Parameters
    ----------
    x : array_like
        cartesian coordinates
    Returns
    -------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.
    """
    
    'Normalize input vector '
    

    x, y, z = cor_car[:,0],cor_car[:,1],cor_car[:,2]
    theta = np.arccos(z / 1)
    phi = np.mod(np.arctan2(y, x), np.pi*2)
    return np.array([np.rad2deg(phi),np.rad2deg(theta)])

def Vis_Spherical_Voxel(Voxel_data,Euler_Angels,Max_Response_num):

    sphere = get_sphere('repulsion724')
    sphere = sphere.subdivide(2)
    mevals = 0.00045*np.ones((Max_Response_num,3))+0.01*np.eye(Max_Response_num,3)
    Max_Or_index = np.argsort(Voxel_data)[::-1][0:Max_Response_num]
    Max_Euler_Angels = Angles[Max_Or_index,:]
    Max_Response_Array = Voxel_data[Max_Or_index]
    fractions = Max_Response_num*[50]
    #angles = [(0, 0), (30, 0)]
    odf = multi_tensor_odf(sphere.vertices, mevals, Max_Euler_Angels, Max_Response_Array)

    # Enables/disables interactive visualization
    interactive = False

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)

    odf_actor = actor.odf_slicer(odf[None, None, None, :], sphere=sphere)
    odf_actor.RotateX(90)
    scene.add(odf_actor)

    print('Saving illustration as symm_signal.png')
    window.record(scene, out_path='symm_signal.png', size=(300, 300))
    #if interactive:
    window.show(scene)