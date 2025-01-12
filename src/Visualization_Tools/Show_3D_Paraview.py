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

# Step 5: Save the visualization (optional)
# Uncomment the following line if you want to save the screenshot
# SaveScreenshot("output_image.png")

# Step 6: Interact with the viewer (this will open the ParaView GUI to view the data)
# (This step is optional as it is useful if you want to continue interacting with the visualization)
# You can leave it running to keep the GUI open for further manipulation.
interact()
