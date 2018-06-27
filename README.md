# Map_Room_Classification
Project by Paul Asquin for Awabot - May-June 2018 paul.asquin@gmail.com  

## I.Introduction
This project automates datasets downloading from ScanNet and Matterport, files treatement, to produce slices from ply files with the different architectures,
and Inception retraining to implement a Room Classification application thanks to Machine Learning.  


## II.Quickstart

This section present a quick way to start the Map Room Classification project.  
With those insutrctions, you will download and transform ScanNet and Matterport Dataset, and retrain Inception with those datas.

    ### 1.Choose ScanNet dataset files with downloading new .txt files
We present ScanNet in script explaination.  

- Go to : https://dovahkiin.stanford.edu/scannet-browse/scans/scannet/querier  
- Display wanted scenes with the search bar (for example : enter "Kitchen")  
- Click on "Save Ids" (the drop-down menu should be on "original")  
- A window opens. Make a Ctrl+S
- Rename the file as "room.txt" (example : kitchen.txt, bedroom.txt, living.txt...)
- Save the .txt file in a folder "Scannet_IDs", at the project root

	### 2.Download, process and retrain
In order to launch the quick "startover" script, open a shell terminal and execute : 
```
sudo python3 big_main.py
```

## III.Script explaination

In this section, we will generally explain the functionnement of our scripts.  
More precise informations are given in the script functions.  

	### 1.Scannet

[ScanNet](http://www.scan-net.org/) is a dataset developped by Stanford University, Princeton University and the Technical University of Munich.  
It consits in the RGB-D scan of multiple rooms, reconstructed in [PLY files](https://en.wikipedia.org/wiki/PLY_(file_format)) (Polygon File Format).  
We have, for each room of a unique type, a PLY file corresponding (a file for a Kitchen, a Bedroom, a Bathroom etc).  

		#### a.Download ScanNet  

In order to smart-download the ScanNet dataset, we have to get the room IDs we want to download. In order to do so, you can follow instructions given in II.1 Choose ScanNet dataset files with downloading new .txt files.  

The script [scannet_download_from_txts.py](scannet_download_from_txts.py) will read those txt files and use the script [scannet_donwload.py](scannet_donwload.py) to download only the PLY file of requested rooms. 
We can notice that their is not only one ply file per room ID (also call scene ID). Here, we are using \_2.ply files, that are at lower resolutions that regulation .ply files.
Because we are going to reduce the resolution of our slices, this is acceptable and will improve our downloading and processing time.
The dataset is written into the "Scannet_PLY" folder and can use up to 220Go. Each type of scene is split in a named folder corresponding to the name of txt files ("Kitchen", "Bathroom").

``` 
python3 scannet_download_from_txts.py
```

		#### b.Process Scannet to slices  

The script [scanner_slicer.py] will obtain the PLY files from the ScanNet dataset (written

``` 
sudo python3 scannet_slicer.py
```

	### 2.Matterport  

		#### a.Download Matterport 

```
python download_mp.py -o HOUSE_SEGMENTATION --type house_segmentations
```
https://github.com/niessner/Matterport

		#### b.Process Matterport to slices  

```
sudo python3 matterport_slicer.py
```

### 3.Inception retraining  

```
sudo python3 dataset_inception_retrain.py
```
