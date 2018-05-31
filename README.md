# scannet_downloader
Project by Paul Asquin for Awabot - May 2018 paul.asquin@gmail.com  

## Introduction
This project automates scene downloading of the ScanNet dataset usiing the download-scannet.py script.

The script download_from_txts.py reads "any.txt" files in the same folder as itself and downloads given scene_ids in the folder "Any".

## Download new .txt files
> Go to : https://dovahkiin.stanford.edu/scannet-browse/scans/scannet/querier  
> Display wanted scenes with the search bar  
> Click on "Save Ids" (the drop-down menu should be on "original")  
> A window opens. Make a Ctrl+S
> Save the .txt file next to the .py scripts

## Usage
``` 
python3 download_from_txts.py
 ```