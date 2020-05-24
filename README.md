# Group 31: Phenotypic Sorting of Cancer Heterogeneity
__Team Members:__ Enrique Carrillosulub, Aswhin Ganesh, Yuko Koike, Shitian Li, Zhe Wei \
__Institution:__ University of California, San Diego\
__Mentor:__ Kevin Chen\
__PI:__ Dr. Stephanie Fraley

## Program Description: 
Our project wants to achieve the automation of phenotypic cell sorting, cell contouring, and photoconversion 
in a 3D-cell culture. Nowadays, 3D-cell cultures are becoming the new standard in cell culture due to the better 
representation of the in vivo environment. On the other hand, the tools for 3D-cell culture are still in 
development. Researchers are still manually sorting, outlining, and photo-converting the cells, which takes 
hours or even days to finish. We are taking steps to automate this process so that researchers can spend their 
time on other tasks. Also, we hope to identify new phenotypes and understand the underlying mechanisms of cancer
migration and progression. We wrote a Python program that can be initiated on the command line to meet these goals. 

## Requirements
* Windows 10 OS
* Python 3.6 installed (Added to Windows Path)
* Nikon NIS Elements Software

## Set up/Installation: 
You can install all necessary Python packages by starting the setup.cmd in this repository, which include the 
following packages: 

* opencv-python
* opencv-contrib-python
* pandas
* xlrd
* nympy
* nd2reader
* datetime
* tqdm

## Instructions: 
1. After imaging your collagen gel culture with Nikon Elements, export your z-stacks as an __nd2__ file and
the coordinates of each z-stack. Save them in a convenient folder on your computer. 
2. To initiate cell segmentation, open up your Command Prompt and change your directory to the folder where 
the __cell_segmentation.py__ is located at. You can do this by typing the following in the Command Prompt: 

```
cd "folderPath"
```

3. To start __cell_segmentation.py__, it needs two required inputs and one optional input: the path to the 
the nd2 file, the path to the excel file of the z-stack coordinates, and a boolean statement indicating if you 
want to save the contoured images and the metrics of each 3D object. The output will be two folders with
the macro files and the binary images of ideal spherical target cells that the Nikon confocal microscope software can 
use to move the microscope to the target cell and draw a region of interest (with binary image). __start_segmentation.cmd__
was written for your convenience. The syntax of using our Python script can be used by typing the following into the 
Command Prompt: 

```
py cell_segmentation.py "nd2_file_path" "coordinates_path" --saveImages True
```

The __first input__ is the path to the nd2 file. The __second input__ is the coordinates of the z-stacks (Excel spread sheet). 
The last paramter is an optional input if you want to save the 3D object cell metrics as a csv fileand images of all
the contours that were detected. The csv file has the following information in order of the columns: 3D object number 
(blob number), z-stack at which its contours are located, contour area, circularity, solidity, and elongation. To 
not have this optional output, omit the --saveImages input

4. Moreover, an example nd2 file with its corresponding coordinates are supplied as an example so clicking on the 
start_segmentation.cmd will initiate the software using the example (which has the --saveImages set to True). 

