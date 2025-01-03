import numpy as np
import sys
import os
import argparse
import importlib
sys.path.append('../oct/pipeline/')
import vol_viz_OCTA
from vol_viz_OCTA import OCTScan, HeaderInfo,BScanHeaderInfo
from vol_viz_OCTA import draw
importlib.reload(vol_viz_OCTA)

def overwrite(fileobj, start, end, newbytes):
    if len(newbytes) != end - start:
        raise ValueError('overwrite cannot expand or contract a file')
    fileobj.seek(start)
    fileobj.write(newbytes)  


' Convert Preproccessed Volume in npy format into modified .vol file  '

parser = argparse.ArgumentParser(description=' Writeout to Vol ')

parser.add_argument("OCTA_npy",
    type=str,
    help="string_path_to_Volume_in_npy_format")

parser.add_argument("Vol_Save_Path",
    type=str,
    help="string_path_to_Volume_in_vol_format_which_will_be_overwritten")

parser.add_argument("Vol_File",
    type=str,
    help="string_path_to_original_Vol_File")
parser.add_argument("Cut_top",
    type=int,
    help="Cut_top")
parser.add_argument("Cut_bottom",
    type=int,
    help="Cut_bottom")
args = parser.parse_args()

Mod_vol = np.load(args.OCTA_npy)

' Read Volume .vol '

file = open(args.Vol_File, "r+b").read()
oct = OCTScan(file)
oct.filename = args.Vol_File.split(os.path.sep)[-1]

' Open Volume File to store the modified data '

file_1 = open(args.Vol_Save_Path, "r+b")

SizeX = oct.headerinfo.SizeX
NumBScans = oct.headerinfo.NumBScans
SizeZ = oct.headerinfo.SizeZ

Coord = np.zeros((3,2),dtype=np.int32)

Coord[0,0] = args.Cut_top
Coord[0,1] = args.Cut_bottom

' The cut in Y and X direction is performed equaly form each side '

if (SizeX-Mod_vol.shape[1])%2 == 0:
        
    Coord[1,0] = int((SizeX-Mod_vol.shape[1])/2)

    Coord[1,1] = SizeX - int((SizeX-Mod_vol.shape[1])/2)

elif (SizeX-Mod_vol.shape[1])%2 == 1:

    Coord[1,0] = int((SizeX-Mod_vol.shape[1])/2)

    Coord[1,1] = SizeX - int((SizeX-Mod_vol.shape[1])/2)+ 1 

if (NumBScans-Mod_vol.shape[2])%2 == 0:
        
    Coord[2,0] = int((NumBScans-Mod_vol.shape[2])/2)

    Coord[2,1] = NumBScans - int((NumBScans-Mod_vol.shape[2])/2)

elif (NumBScans-Mod_vol.shape[2])%2 == 1:

    Coord[2,0] =np.rint((NumBScans-Mod_vol.shape[2])/2)

    Coord[2,1] = NumBScans-np.rint((NumBScans-Mod_vol.shape[2])/2) + 1 

HEADERSIZE  = 2048
BHEADERSIZE = 256
header      = file[:HEADERSIZE]

' Read data after Headerinfo '

offset = HEADERSIZE

' Read all the Data_fields from .vol file '

headerinfo = HeaderInfo(header)

' Define Step Size to iterate over entire Bscan '

slo_size = headerinfo.SizeXSlo * headerinfo.SizeYSlo

' read SLO image (byte format) '

SLO = np.frombuffer(file, dtype='B', count=slo_size, offset=offset).reshape((headerinfo.SizeXSlo, headerinfo.SizeYSlo))
offset += slo_size

' Get_Current_Bscan_pos '

for Bscan in range(Coord[2,0]):
            
    # read header:
        
    BsBlkSize = headerinfo.BScanHdrSize + headerinfo.SizeX * headerinfo.SizeZ * 4
    bheader = BScanHeaderInfo(file[offset:offset + BHEADERSIZE])
    offset += BHEADERSIZE
    
    ' Get Position of Raw BScan_Data '
        
    offset += bheader.BScanHdrSize - BHEADERSIZE
        
    ' Iterate over normalized BScans '
        
    count_B_Scan    = headerinfo.SizeX * headerinfo.SizeZ 
        
    ' Define new offset to begin with the next scan '

    offset += headerinfo.SizeX * headerinfo.SizeZ * 4 # 4 stands for 4 bytes of float format

' Iterate over BScans '

for k,Bscan in enumerate(range(Coord[2,0],Coord[2,1])):
            
    # read header:
        
    BsBlkSize = headerinfo.BScanHdrSize + headerinfo.SizeX * headerinfo.SizeZ * 4
    bheader = BScanHeaderInfo(file[offset:offset + BHEADERSIZE])
    offset += BHEADERSIZE
    
    ' Get Position of Raw BScan_Data '
        
    offset += bheader.BScanHdrSize - BHEADERSIZE
        
    ' Iterate over normalized BScans '
        
    count_B_Scan    = headerinfo.SizeX * headerinfo.SizeZ 
    
    ' Write to File '
    
    Copy_Data = np.frombuffer(file, dtype=np.float32, offset=offset,count=headerinfo.SizeX * headerinfo.SizeZ).reshape((headerinfo.SizeZ, headerinfo.SizeX)).copy()

    Copy_Data[Coord[0,0]:Coord[0,1],Coord[1,0]:Coord[1,1]] = Mod_vol[:,:,k]

    OCT_Vol_bytes = Copy_Data.tobytes()
    
    overwrite(file_1,offset,offset + headerinfo.SizeX * headerinfo.SizeZ * 4, OCT_Vol_bytes)
        
    ' Define new offset to begin with the next scan '
        
    offset += headerinfo.SizeX * headerinfo.SizeZ * 4 # 4 stands for 4 bytes of float format 
        