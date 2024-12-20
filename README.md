# CAE
This is the official code repository for class association embedding (CAE) learning and topology analysis. We also present one case for samples generation using one trained CAE model here.


## 0. Main Environments
```bash
conda create -n CAE python=3.9
conda activate CAE
pip install torch==1.12.1
pip install torchvision==0.13.1
pip install numpy==1.24.4
pip install pandas==1.3.5
pip install kmapper==2.1.0
pip install scikit-learn==1.0.2
pip install matplotlib==3.5.1
pip install pillow==9.0.1
pip install opencv-python==4.10.0.84
pip install opencv-python-headless==4.10.0.84
pip install openpyxl==3.1.5
pip install networkx==3.1
pip install argparse==1.1
```



## 1. Prepare the dataset

## Data used for training and testing:
The data used for training and testing can be downloaded from https://drive.google.com/drive/folders/1g2RnUZZmBhmBi_IzplgT9sN-Nz2Byr67?dmr=1&ec=wgc-drive-hero-goto

## Explanation Regarding Training and Testing Datasets
- After downloading the datasets, you are supposed to put them into './data/', and the file format reference is as follows. (take the Brain Tumor2 dataset as an example.)

- './data/Brain Tumor2/'
  - trainA_img
    - .png
  - trainB_img
    - .png
  
  - testA_img
    - .png
  - testB_img
    - .png

- training images with normal label will be put into 'trainA_img/' folder, while training images with abnormal labels will be put into 'trainB_img/'
- test images with normal label will be put into 'testA_img/' folder, while test images with abnormal labels will be put into 'testB_img/'
- the names and the labels of the training images (with the format 'image_name label') are put into the 'trainAB_img-name_label.txt'
- the names and the labels of the test images (with the format 'image_name label') are put into the 'testAB_img-name_label.txt'
- '0' represents normal class label while other numbers represent abnormal classes in our work

## 2. Train the CAE
Open the 'CAE_train' folder and run the 'main_train.py' file, then the CAE model starts training
```bash
cd code/CAE_train
python main_train.py  # Train and test CAE model.
```

## 3. Obtain the trained CAE models and generated results of some cases
- After trianing, you could obtain the trained models in '/results/models/'.

  The trained CAE models that are used for extracting class-associated codes and generating new samples are named 'gen_index.pt' (where 'index' represents the iteration number of training).

- After trianing, you could obtain the generated results of some cases in '/results/images_display/'.

  In the '/results/images_display/' folder, for each image file, the first row is the original cases, the second row presents the samples generated by combining the class-associated codes and the individual codes from the original cases, the third row is the donor samples whose class-associated codes are extracted, and the fourth row presents the samples generated by combining the individual codes from the original samples in the first row with the class-associated codes from the samples in the third row.
  
  In the '/results/images_display/' folder, the file name, for example, 'gen_a2b_test2_00104000.jpg', means the generated results from A class (normal) to B class (abnormal) on test2 group samples using trained CAE model which has been trained 104000 iterations.


## 4. Perform class-associated codes analysis on test datasets
Open the 'topology_analysis' folder
```bash
cd code/topology_analysis
```

Run the 'CL_codes_extract.py' file, so we can extract class-associated codes of the test images using trained CAE model, which is put into the './code/trained_models/' folder.
The class-associated codes extracted from the test dataset are put into the './code/topology_analysis/results/testAB_CL_codes_extraction_results.csv' file. Each code consists of 8 values.
```bash
python CL_codes_extract.py  # Extract class-associated codes of the test images using trained models.
```

Run the 'topological_analysis.py' file, so we can perform topological analysis on extracted class-associated codes.
The topological data analysis results are put into the './code/topology_analysis/results/topological_analysis_custom_image_name_result.html' file and the './code/topology_analysis/results/topological_analysis_custom_labels_result.html' file.(these two files present the same topological graph result, except that one uses the image names to represent samples, while the other use the class labels to mark samples)
```bash
python topological_analysis.py  # Perform topological analysis on extracted class-associated codes.
```



## 5. Perform instance explanation using the class-associated manifold
- After performing topological analysis on extracted class-associated codes, we can get one topological graph, and the images involved into each node within the graph should be recorded (format: ID_number image1_name image2_name...) into the 'Nodes_images_name.txt' file.
- We can follow these steps for performing samples generation (generated samples can be also used for tumor segmentation task) and instance explanation:

5.1 Open the 'code/Case_Show/' folder.
```bash
cd code/Case_Show
```

5.2 Run the 'graph_nodes_distance_matrix.py' file to calculate the distances between each two nodes within the topological graph obtained from the class-associated codes, and output one distance matrix representing the relations between nodes.
The distance matrix is put into the './code/Case_Show/results/Nodes_connection_distance.csv' file, in which nodes are marked as 'ID_number', and null value represents that these two nodes are not connected. 
```bash
python graph_nodes_distance_matrix.py  # Calculate the distances between each two nodes within the topological graph obtained from the class-associated codes, and output one distance matrix representing the relations between nodes.
```

5.3 Run the 'shortest_path_get_for_each_two_points.py' file to get the shortest path between two samples within the topological graph.
For one instance to be explained, we select one counter reference sample for it, then the starting and the ending nodes could be obtained by calculating and selecting nodes with lowest distances to the explained sample and the counter reference sample, respectively.
And we calculate the shortest path between the starting node and the ending node based on the nodes distance matrix.
The shortest path between the starting node and the ending node within the graph is recorded into the './code/Case_Show/results/AB_images_shortest_path_save.csv' file.
```bash
python shortest_path_get_for_each_two_points.py  # For one instance to be explained, we select one counter reference sample for it, and calculate the shortest path between these two sample based on the nodes distance matrix.
```

5.4 Run the 'local_explanation_on_instance.py' file to generate saliency map for instance explanation.
Along the shortest path obtained as mentioned above, we obtain meaningful class-associated codes for guided counterfactual generation, and by analyzing the changes of the generated samples and the changes of the outputs of the black-box model on the generated samples, we can get one saliency map for the instance explanation.
The generated samples are put into the './code/Case_Show/results/generate_img_saliency_maps_for_local_explanation/' folder. For example, 'ex_z_86_01567.png_ref_z_86_01318.png_ID_0_gen.png' is the generated sample obtained by combining the individual code of the explained sample 'z_86_01567.png' and the class-associated code obtained by calculating the codes center of the starting node, and here the 'ref_z_86_01318.png' indicates that the counter reference image is z_86_01318.png, which is used to calculate the shortest path (end point).
The saliency map obtained for instance explanation is also put into the './code/Case_Show/results/generate_img_saliency_maps_for_local_explanation/' folder.
The generated samples can be also used to obtain the tumor segmentation result by comparing with the original sample. 
```bash
python local_explanation_on_instance.py  #  Along the shortest path, we obtain meaningful class-associated codes for guided counterfactual generation, and by analyzing the changes of the generated samples and the changes of the outputs of the black-box model on the generated samples, we can get one saliency map for the instance explanation. 
```


