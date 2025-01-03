import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.util import numpy_support
from pyevtk.hl import gridToVTK

def numpy_to_image(v_np):
    """Convert a numpy 2D or 3D array to a vtkImageData object.

    v_np
        2D or 3D numpy array containing image data

    return
        vtkImageData with the v_np content
    """
    shape = v_np.shape
    h, w = shape[0], shape[1]
    c = 1
    if len(shape) == 3:
         c = shape[2]
    # Reshape 2D image to 1D array suitable for conversion to a
    # vtkArray with numpy_support.numpy_to_vtk()
    linear_array = np.reshape(v_np, (w*h, c))
    vtk_array = numpy_support.numpy_to_vtk(linear_array)
    image = vtkImageData()
    image.SetDimensions(w, h, 1)
    image.AllocateScalars(vtk_array.GetDataType(), 4)
    image.GetPointData().GetScalars().DeepCopy(vtk_array)
    return image

def save_to_vtk(data, filepath):
        """
        save the 3d data to a .vtk file. 
        
        Parameters
        ------------
        data : 3d np.array
                3d matrix that we want to visualize
        filepath : str
                where to save the vtk model, do not include vtk extension, it does automatically
        """
        x = np.arange(data.shape[0]+1)
        y = np.arange(data.shape[1]+1)
        z = np.arange(data.shape[2]+1)
        gridToVTK(filepath, x, y, z, cellData={'data': data.copy()})

from tvtk.api import tvtk, write_data
import numpy as np

' Save 3_D vector field  '

def Save_to_vtk_4D(data,filepath):
    grid = np.ones((data.shape[0:3]))
    i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
    i.point_data.scalars = grid.ravel()
    i.point_data.scalars.name = 'Voxels'
    i.dimensions = grid.shape[0:3]
    # add second point data field
    i.point_data.add_array(data.ravel())
    i.point_data.get_array(1).name = 'Orientation_Data'
    i.point_data.update()

    write_data(i, filepath+'vtktest.vtk')



def convert_npy_to_vtk(input_npy, output_vtk):
    """
    Convert a 3D NumPy array stored in a .npy file to a .vtk file for visualization in ParaView.
    
    Parameters:
        input_npy (str): Path to the input .npy file.
        output_vtk (str): Path to the output .vtk file.
    """
    # Load the numpy array
    volume = np.load(input_npy)

    # Check the dimensions
    print(volume.shape)
    if len(volume.shape) != 3:
        raise ValueError("The numpy array must be 3D.")

    # Create a VTK image data object
    vtk_image = vtk.vtkImageData()

    # Set dimensions of the VTK image
    vtk_image.SetDimensions(volume.shape[::-1])  # VTK expects z, y, x ordering
    vtk_image.SetOrigin(0, 0, 0)  # Origin of the volume
    vtk_image.SetSpacing(1, 1, 1)  # Spacing of the grid (modify if needed)

    # Convert the numpy array to VTK data
    vtk_data = numpy_to_vtk(num_array=volume.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data.SetName("Scalars")  # Name of the scalar field

    # Attach the data to the VTK image
    vtk_image.GetPointData().SetScalars(vtk_data)

    # Write the VTK file
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(output_vtk)
    writer.SetInputData(vtk_image)
    writer.Write()

    print(f"Converted {input_npy} to {output_vtk}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='OCTA Volume Preprocessing Routine for Vessel Enhancement')
    parser.add_argument("vol_np",
            type=str,
            help="Path to numpy volume")
    parser.add_argument("--output",
        type=str,
        help="Output filename",
        default=None)
    args = parser.parse_args()

    # specify output filepath
    assert args.vol_np.endswith(".npy")
    if args.output is None:
        out_fpath = args.vol_np[:-4]
    else:
        out_fpath = args.output
        if args.output.endswith(".vtk"):
            out_fpath = args.output[:-4]

    # convert volume to vtk file
    v_np = np.load(args.vol_np)
    save_to_vtk(v_np, out_fpath)

'''if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert a 3D NumPy array (.npy) to a VTK file (.vtk) for ParaView.")
    parser.add_argument("input_npy", type=str, help="Path to the input .npy file.")
    parser.add_argument("output_vtk", type=str, help="Path to the output .vtk file.")

    args = parser.parse_args()

    # Run the conversion function
    convert_npy_to_vtk(args.input_npy, args.output_vtk)'''
