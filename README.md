#room_classification_from_scratch_cnn
Project by [Paul Asquin](https://www.linkedin.com/in/paulasquin/) for Awabot - Summer 2018 paul.asquin@gmail.com  

# I.Introduction  
This repo is a part of the Room Classification Project. 
The aim of the Room Classification Project is to make an indoor mobile robot able to recognize a room using its 2D map. 
The output of the 2D map given should be "Kitchen", "Bedroom", "Batroom", etc.  

In order to achieve this goal, we have chosen to use Machine Learning techniques in order to obtain a powerfull recognition system with no hard-coded rules.  

As for every Machine Learning project, we need adapted datasets and a learning algorithm.  

Here is the overall architecture of the project :   
.  
├── room_classification_get_datasets  
├── room_classification_from_scratch_cnn  
├── room_classification_network_retrain  
├── Datasets (created be room_classification_get_datasets)  

Before comming to this repo, you should have installed the project datasets using _room\_classification\_get\_datasets_

# II. Goals and instructions
The goal of this repo is to use pretrained image recognition models, like inception_v3 or pnasnet, in order to build a new model for our project of room classification.   

At this step, you may have already installed the dependencies with _room\_classification\_get\_datasets_.  

# 1. Retrain a model

In order to retrain a model, you have to set wich model you want to use. For this, go into the file [dataset_retrain.py](dataset_retrain.py) and add in the _cmd_ string, the argument ' --tfhub_module 'urlToTheWantedModel'.
You can find such url in [this page](https://www.tensorflow.org/hub/modules/image) : click on the model you want and the given **Module URL**. Also, be sure that _\-\-image-dir_ is targeting your dataset.  
_
If you want too, you can change other parameters as --validation_percentage, --validation_batch_size... You will find the explaination of their influence in [this tutorial](https://www.tensorflow.org/hub/tutorials/image_retraining).  

Those parameters will be used during the call of [retrain.py](retrain.py), a script by Google. Still, during the developement, we also have added personnal argument to [retrain.py](retrain.py) for our personnal use.  
Here are the argument we have added :  
\- suffix (Optional) ; String: Consider only particular suffix in file names.  
\- \-\-in_name (Optional) ; String: Consider only particular file names containing the str in_name in their name.  
\- \-\-augmentation (Optional, Default: False) ; Bool: If True, Augment the dataset, if False choose a random given transformation of a room slice.  
\- \-\-train_maximum (Optional, Default: False) ; If False, do nothing. If True, continue the program waiting for a CTRL+C, catched before making a clean exit.  
\- \-\-path_mislabeled_names (Optional, String) ; Path where to write a txt file containing names of mislabeled pictures.  