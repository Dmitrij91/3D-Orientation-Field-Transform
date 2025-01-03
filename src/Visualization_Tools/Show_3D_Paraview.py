# Import ParaView's simple module
from paraview.simple import *
import argparse 

parser = argparse.ArgumentParser(description='Visualize Volume.vtk data')

parser.add_argument("Path",
    type=str,
    help="Path_to_Volume_in_npy_format")
args = parser.parse_args()

# Step 1: Load the .vtk file
file_path = args.Path  # Update this with the path to your .vtk file
vtk_data = OpenDataFile(file_path)

# Step 2: Apply the data
vtk_data.UpdatePipeline()

# Step 3: Set up visualization
# Optionally, set the color map and representation (surface, wireframe, etc.)
vtk_data_rep = Show(vtk_data)
vtk_data_rep.ColorArrayName = ['POINTS', 'your_array_name']  # Replace 'your_array_name' with the data array name if needed

# Step 4: Render the view
Render()
