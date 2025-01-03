import numpy as np
import argparse
import os.path
import sys
import vol_viz_OCTA
from vol_viz_OCTA import OCTScan
from vol_viz_OCTA import draw


parser = argparse.ArgumentParser(description='Load OCTA volume and save as npy_file')

parser.add_argument("OCTA_File",
    type=str,
    help="Path to OCTA .vol file")
parser.add_argument("--window",
    type=int,
    help="Rectangular_Dimension_of_Cropped_Volume",
    default=400)

args = parser.parse_args()
assert os.path.isfile(args.OCTA_File), f"File {args.OCTA_File} not found."
assert args.OCTA_File.endswith(".vol")   

file = open(args.OCTA_File, "rb").read()
oct = OCTScan(file)
oct.filename = args.OCTA_File.split(os.path.sep)[-1]

'Load OCTA Data'

X = oct.headerinfo.SizeX
Y = oct.headerinfo.NumBScans
Z = oct.headerinfo.SizeZ

assert args.window <= np.min(np.array([X,Y,Z])), " Cropped Dimension exceeds volume dimensions "

octdata_full  = np.zeros((Z,X,Y)) 
segments_full = np.zeros((oct.bscans[0].segments.shape[0], X, Y))
for k in range(Y):
    octdata_full[:,:,k] = ((oct.bscans)[k]).data
    segments_full[:,:,k] = ((oct.bscans)[k]).segments
    
    ' Remove_Scanner_Artifacts '

octdata_full[octdata_full > 10] = 1e-10
octdata_full[octdata_full <= 0] = 1e-10

'Get Layer Mask'

SizeZ, SizeX, NumBScans = octdata_full.shape
layer_mask = np.zeros((Z,X,Y))

for b in range(Y):
    seglines = segments_full[:,:,b]

    min_seglines = np.min(seglines, axis=1)
    idx = np.argsort(min_seglines)
    seglines = seglines[idx, :]

    bscan_mask = np.zeros((SizeZ,SizeX))
    sline = 1
    for k in range(seglines.shape[0]-1):
        boundary1 = np.round(seglines[k][1:-2]).astype(int)
        # boundary2 = np.round(seglines[k + 1][1:-2]).astype(int)

        if not np.any(boundary1 < 0):
            for j in range(1,len(seglines[0])-3):
                bscan_mask[boundary1[j]:,j] = sline

            sline += 1

    layer_mask[:,:,b] = bscan_mask


' Crop Volume '

if (X-args.window)%2 == 0:

    Cut_Borders_X = (int((X-args.window)/2),int((X-args.window)/2))

elif (X-args.window)%2 == 1:

    Cut_Borders_X = (np.rint((X-args.window)/2),np.rint((X-args.window)/2)+1)

octdata_full = octdata_full[:,Cut_Borders_X[0]:X-Cut_Borders_X[1],:]
layer_mask = layer_mask[:,Cut_Borders_X[0]:X-Cut_Borders_X[1],:]


if (Y-args.window)%2 == 0:

    Cut_Borders_Y = (int((Y-args.window)/2),int((Y-args.window)/2))

elif (Y-args.window)%2 == 1:

    Cut_Borders_Y = (np.rint((Y-args.window)/2),np.rint((Y-args.window)/2)+1)

octdata_full = octdata_full[:,:,Cut_Borders_Y[0]:X-Cut_Borders_Y[1]]
layer_mask = layer_mask[:,:,Cut_Borders_Y[0]:X-Cut_Borders_Y[1]]

if (Z-args.window)%2 == 0:

    Cut_Borders_Z = (int((Z-args.window)/2),int((Z-args.window)/2))

elif (Z-args.window)%2 == 1:

    Cut_Borders_Z = (np.rint((Z-args.window)/2),np.rint((Z-args.window)/2)+1)

octdata_full = octdata_full[Cut_Borders_Z[0]:Z-Cut_Borders_Z[1],:,:]
layer_mask   = layer_mask[Cut_Borders_Z[0]:Z-Cut_Borders_Z[1],:,:]

' Save Volume '

np.save(os.path.join("Data_Folder/","octdata_full_numpy"),octdata_full)
np.save(os.path.join("Data_Folder/","octdata_full_numpy_mask"),layer_mask)