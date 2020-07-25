# Single Image Surface Appearance Modeling with Self-augmented CNNs and Inexact Supervision

The main contributors of this repository include Wenjie Ye,  [Xiao Li](http://home.ustc.edu.cn/~pableeto), [Yue Dong](http://yuedong.shading.me), [Pieter Peers](http://www.cs.wm.edu/~ppeers/) and [Xin Tong](https://www.microsoft.com/en-us/research/people/xtong/).

## Introduction

This repository provides a reference implementation for the PG 2018 paper "Single Image Surface Appearance Modeling with Self-augmented CNNs and Inexact Supervision".

More information (including a copy of the paper) can be found at http://msraig.info/~InexactSA/inexactsa.htm.

## Citation
If you use our code or models, please cite:

```
@article{Ye:2018:SIS, 
 author = {Ye, Wenjie and Li, Xiao and Dong, Yue and Peers, Pieter and Tong, Xin},
 title = {Single Image Surface Appearance Modeling with Self-augmented CNNs and Inexact Supervision},
 year = {2018},
 journal = {Computer Graphics Forum},
 volume = {37},
 number = {7},
 pages = {201--211},
 }
```

----------------------------------------------------------------
## Usage

### System requirements
   - Linux system (tested on Ubuntu 16.04).
   - An NVidia GPU (tested on Titan X).
   - CUDA toolkit (tested with version 8.0).
   - Tensorflow (tested with version 1.4.1).
   - Python 3 (tested with version 3.5.2). The following packages are required:
     * OpenCV
     * numpy
     * transforms3d
     * PIL
     * skimage
   - gcc (tested with version 5.4.0). The following packages are required:
     * glm
     * OpenCV

### Docker Image
For convenience test purpose, we provide a docker image with configured environment and data. 
To pull the docker image, please run

    docker pull pableeto/inexact_sa:tf-1.4

and you are ready to run all the test, training and empirical study experiments inside this docker image.
To activate the docker environment, please run (add 'sudo' before docker command if you get 'permission denied' error):
    
    docker run --gpus=all --ipc=host --net host -it pableeto/inexact_sa:tf-1.4 bash

We have already prepared all the data, models and scripts for reproduce our results in /InexactSA_code under the docker image. 
see below for how to reproduce our experiments.
You could also setup the environment by yourself, please follow the instruction below.

### Preparations
##### Compile TensorFlow pixel shader. 
It is a TensorFlow op provided by Xiao Li.

Open the file "./render/compile_kernel_fast.sh", in the 5th line, input the absolute directory for glm; in the 6th line, input the absolute directory of the code; in the 7th line, input the absolute directory of the CUDA library. 

The code also supposes that TensorFlow is installed in a virtual environment named "tensorflow" created by virtualenv. If this does not fit your case, you also need to modify the first line according to your TensorFlow installation. 

Run the compiling script, and get compiled file "render_all_fast.so".

    cd render
    ./compile_kernel_fast.sh
    mv render_all_fast.so ../
    cd ..
    
##### Download data. 
Related data can be found at our [project page](http://msraig.info/~InexactSA/inexactsa.htm), including:
- envMaps.zip: Environment maps and lighting settings, needed if you want to run any training. 
- InputImages.zip: 4827 input images for training the network for real world materials.
- SynthesisOutput.zip: The neural texture synthesis results of the 4827 input images.
- TestData_real.zip: Real world material test data with artist-labeled ground truth. 
- model.zip: The trained network for real world materials.
<!-- - FreeMono.ttf: A font file, used in test result output during training. It also could be found on your computer.  -->

In the following instructions, we suppose all these files are stored in ./resources. 

Unzip the environment maps if you want to run training. 

    unzip ./resources/envMaps.zip

### Test a model on unlabeled images
- If you want to use the pretrained model, unzip it.

      unzip ./resources/model.zip -d ./pretrained_model

- Put the images to test (png or jpg) in a folder, eg. ./data/real/test_unlabeled. The suggested resolution is 256*256. 
- Run the test.

      python test.py --loadModel ./pretrained_model/model.ckpt-100000 --input_folder ./data/real/test_unlabeled --output_folder ./output/test_unlabeled

Please note that the meanings of the input for all programs can be found in the source code, and will also be printed when the program is given an invalid input, so they will not be detailed here. 

You can also use a model trained by yourself; Please refer to next section for training our model. 

### Train a model for real world materials
##### Compile the c++ code for converting diffuse maps to normal maps. 
OpenCV is needed for the compilation. We use OpenCV version 4.0. If you use another version, please modify the command line according to your installation. 

    g++ ./convertDiffuseToNormal/genSynData.cpp ./convertDiffuseToNormal/pfm.cpp -o convert.so -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -std=c++11
    
Note: For compatibility, in all provided c++ code, the output folder will NOT be created automatically if it do not exist, so you need to create them by yourself. 

##### Prepare the input images.
They will also serve as unlabeled photographs for self-augmentation. 

    unzip ./resources/InputImages.zip -d ./data/real
    
##### Generate labeled data. 
- Generate diffuse maps. We apply neural texture synthesis on the input images to get labeled diffuse maps. We use the [implementation](https://github.com/leongatys/DeepTextures) of [Gatys et al. 2015]. Note we only use the first supposed layers in the computation of Gram matrix (i.e., conv1_1, pool1, pool2). You can also directly download and use our synthesis result.

      unzip ./resources/SynthesisOutput.zip -d ./data/real
      
- Convert diffuse maps to normal maps. Note: create the output folder by yourself. 

      ./convert.so 4827 ./data/real/SynthesisOutput ./data/real/labeled
      
- Generate specular values.

      python genSpecular.py 4827 ./data/real/labeled
      
##### Prepare the test data. 
We rearranged the dataset released by [Li et al. 2017].
    
    unzip ./resources/TestData_real.zip -d ./data/real
    
##### Train the model. 
Test will be run during the training. The output needs a font file "FreeMono.ttf". If it is not found on your computer, please get it and put it into the program folder. 

Training with 2 GPU cards will be faster than using a single card, while using more cards will not lead to more speed increasement. 

    python train.py --labeled_step 50000 --augment_step 50000 --testFreq 10000 --outputFreq 20 --saveFreq 10000 --labeled_folder ./data/real/labeled --unlabeled_folder ./data/real/InputImages --lighting_folder ./envMaps --test_folder ./data/real/TestData_real --output_folder ./output/real

### Reproduce our empirical study
##### Compile the Perlin data generation code. 
We provide c++ code for generating Perlin textures in our empirical study. 

To compile the c++ executable, run:

    g++ ./genDiffuseNormal/genSynData.cpp ./genDiffuseNormal/pfm.cpp -o gen.so -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc  -std=c++11 -fopenmp

    
##### Study for ambiguity
- Generate labeled data. Note: create the output folder by yourself. 

      ./gen.so 1 5 ./data/ambiguity/labeled
      python genSpecular.py 5000 ./data/ambiguity/labeled

- Generate unlabeled data. We first generate labeled data, and then render them to get unlabeled data.

      ./gen.so 1 10 ./data/ambiguity/labeled_to_gen_unlabeled
      python genSpecular.py 10000 ./data/ambiguity/labeled_to_gen_unlabeled
      python genUnlabeled.py ./data/ambiguity/labeled_to_gen_unlabeled ./data/ambiguity/unlabeled ./envMaps
      
- Generate test data. We shuffle normal maps after generation and fix the order in all training. 

      ./gen.so 1 1 ./data/ambiguity/test
      python genSpecular.py 1000 ./data/ambiguity/test
      python shuffleNormal.py ./data/ambiguity/test
      
- Run the training for regular target. 
      
      python train.py --labeled_step 50000 --augment_step 50000 --testFreq 10000 --outputFreq 20 --saveFreq 10000 --labeled_folder ./data/ambiguity/labeled --unlabeled_folder ./data/ambiguity/unlabeled --lighting_folder ./envMaps --test_folder ./data/ambiguity/test --output_folder ./output/ambiguity/regular
      
- Generate data and train for ambiguity target. 

      python genPerlinAmbi.py ./data/ambiguity/labeled ./data/ambiguity/labeled_ambiguity ./envMaps 1
      python train.py --labeled_step 50000 --augment_step 50000 --testFreq 10000 --outputFreq 20 --saveFreq 10000 --labeled_folder ./data/ambiguity/labeled_ambiguity --unlabeled_folder ./data/ambiguity/unlabeled --lighting_folder ./envMaps --test_folder ./data/ambiguity/test --output_folder ./output/ambiguity/ambiguity
      
- Generate data and train for mixture target. 

      python genPerlinAmbi.py ./data/ambiguity/labeled ./data/ambiguity/labeled_mixture ./envMaps 2
      python train.py --labeled_step 50000 --augment_step 50000 --testFreq 10000 --outputFreq 20 --saveFreq 10000 --labeled_folder ./data/ambiguity/labeled_mixture --unlabeled_folder ./data/ambiguity/unlabeled --lighting_folder ./envMaps --test_folder ./data/ambiguity/test --output_folder ./output/ambiguity/mixture
      
##### Study for inexactness of labeled data
We show an example procedure to train a regular-to-sharp model, and test on sharp data. 

- Generate regular labeled data

      ./gen.so 1 5 ./data/inexact/labeled_regular
      python genSpecular.py 5000 ./data/inexact/labeled_regular
      
- Generate sharp unlabeled data. 

      ./gen.so 3 10 ./data/inexact/labeled_to_gen_unlabeled_sharp
      python genSpecular.py 10000 ./data/inexact/labeled_to_gen_unlabeled_sharp
      python genUnlabeled.py ./data/inexact/labeled_to_gen_unlabeled_sharp ./data/inexact/unlabeled_sharp ./envMaps
      
- Generate sharp test data. 

      ./gen.so 3 1 ./data/inexact/test_sharp
      python genSpecular.py 1000 ./data/inexact/test_sharp
      python shuffleNormal.py ./data/inexact/test_sharp

- Run the training. 

      python train.py --labeled_step 50000 --augment_step 50000 --testFreq 10000 --outputFreq 20 --saveFreq 10000 --labeled_folder ./data/inexact/labeled_regular --unlabeled_folder ./data/unlabeled_sharp --lighting_folder ./envMaps --test_folder ./data/inexact/test_sharp --output_folder ./output/inexact/regular2sharp

## Acknowledgement
Part of the Python code is based on [the code of self-augmented-net](https://github.com/msraig/self-augmented-net). 

The c++ pfm interface is based on [the code from WORD GEMS](https://wordgems.wordpress.com/2010/12/14/quick-dirty-pfm-reader/). 

The c++ Perlin header is from [siv::PerlinNoise](https://github.com/Reputeless/PerlinNoise).

## Contact
You can contact Wenjie Ye (ywjleft@163.com) or Xiao Li (pableetoli@gmail.com) if you have any problems.

## Reference
[1] LI X., DONG Y., PEERS P., TONG X.: Modeling surface appearance from a single photograph using self-augmented convolutional neural networks. ACM Trans. Graph. 36, 4 (July 2017), 45:1–45:11.

[2] GATYS L., ECKER A. S., BETHGE M.: Texture synthesis using convolutional neural networks. In Advances in Neural Information Processing Systems (2015), pp. 262–270.
