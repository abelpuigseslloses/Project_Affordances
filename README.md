**PROJECT AFFORDANCES**

I did this project as part of my Master's thesis. I set out to train Deep Neural Network models to predict scene affordances. For this, I used the Sun Attribute Dataset (Patterson et al., 2014) to train a ResNet50 and ViT Base Patch 16 end-to-end. The goal was to have models that output a list of all possible actions or affordances of a given scene - so they are trained on a multi-label image classification task. The code here allows the training of DNN models on affordances but also the other types of attributes (materials, surface properties...).

In this repo, we find:

*main_script.py*: run it to train the model. I set default parameters here (argparse) and they can be adjusted with the *sweep_config.yaml* file.

*train_eval.py*: the main script calls this function to train the model while testing on the validation set.

*utils.py*: get useful functions for training (creates the Dataset, gets the model, dataloaders, apply data transforms, set the seeds, etc.)

*data*: The files that make up the dataset are the following (Patterson et al., 2014):

	-attributes.mat: A cell array containing all of the 102 attribute words 
	
	-images.mat: A cell array containing all 14340 image names. 
			The image name strings are formatted in exactly the same way as the
			SUN database. In order to access the jpg for a given image, use the 
			following commands in Matlab:
				>> data_path = '{The path where you have save the SUN Images}';
				>> imshow(strcat(data_path, images{#img}));
	
	-attributeLabels_continuous.mat: This file contains a 14340x102 matrix. 
					Each row is an attribute occurance vector. 
					Attribute occurance vectors are vectors of real-valued labels
					ranging from 0-1, which correspond to how often a given
					attribute was voted as being present in a given image
					by Mechanical Turk Workers. These continuous values are 
					calculated from 3 votes given by the AMT workers for each
					image.
					The indicies of this matrix correspond with the cell arrays
					in attributes.mat and images.mat, e.g. the first row
					in 'attribute_labels' is the first image in 'images', and
					the first column in 'attribute_labels' is the first attribute
					in 'attributes'.
*my_env.yaml*: Conda environment YAML file that specifies the configuration and dependencies for Conda environment. In terminal, type *conda env create -f myenv.yml* to recreate the environment.
*sweep_config.yaml*: Configuration to start sweep (used with wandb for tracking). Specify hyperparameters, type of search, etc. here.

