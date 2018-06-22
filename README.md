# Map_Room_Classification
Project by Paul Asquin for Awabot - May-June 2018 paul.asquin@gmail.com  

## Introduction
This project automates datasets downloading from ScanNet and Matterport, files treatement to produce slices from ply files with the different architectures 
and Inception retraining to implement a Room Classification application thanks to Machine Learning.  

The script download_from_txts.py reads "any.txt" files in the same folder as itself and downloads given scene_ids in the folder "Any".

## Datasets Management

### Scannet

#### Download new .txt files
- Go to : https://dovahkiin.stanford.edu/scannet-browse/scans/scannet/querier  
- Display wanted scenes with the search bar  
- Click on "Save Ids" (the drop-down menu should be on "original")  
- A window opens. Make a Ctrl+S
- Save the .txt file next to .py scripts

#### Download the dataset  

``` 
python3 scannet_download_from_txts.py
```

#### Process Scannet to slices  

``` 
sudo python3 scannet_slicer.py
```

### Matterport  

#### Download the dataset  

```
python download_mp.py -o HOUSE_SEGMENTATION --type house_segmentations
```
https://github.com/niessner/Matterport

#### Process Matterport to slices  

```
sudo python3 matterport_slicer.py
```

## Inception retraining  

```
sudo python3 dataset_inception_retrain.py
```
