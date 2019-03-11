# CNN based Autoencoders :
Implementation of CNN based autoencoders.

#### Author: [Ravin Kumar](https://mr-ravin.github.io)

##### Written in Tensorflow: 1.2

## Directory Structure :
```
|
|--input   (dir)
|
|--pros    (dir)
|
|--restmp  (dir)
|
|--test    (dir)
|
|--testres (dir)
|
|--script.py
|
|--run.py
```

### Description of various directories and files :
- input directory contains input images.
- test directory contains test images.
- pros directory will contain processed images, that then feed to the encoder side of autoencders.
- restmp directory contains intermediate results during training phase.
- testres directory will contain the output over the images at inference time.
- script.py this is a module based file in python, where cnn based autoencoders are implimented.
- run.py file is automated version of the process.
- Note: lets say at training time input image name is "abc.jpg", then please save it's desired output image by "abcr.jpg" in the input directory.

### Steps required :
- We can import script.py file and than proceed :
```
import script
script.preprocess() # it will convert all input dir images to fixed size of 256 x 256.
# processed images are saved to pros directory.
script.generate_more_data(90) # it will generate more data from processed images of pros directory.
# by rotating them. here images are rotated by 90 degree to produce more data.
script.run(1000) #  1000 defines the number of epoches for training.
```
- To automate the process of training and testing :
```
>>> python3 run.py 1000
```
Note: you do have to resize the images of test directory manually or by using the script code seprately.

### Working Project Videos
A sample working demonstration of Autoencoders is available at [Youtube](https://www.youtube.com/watch?v=acfAf6eLbh8).

[![working demonstration](https://github.com/mr-ravin/CNN-Autoencoders/blob/master/flag_autoencoders.gif)](https://www.youtube.com/watch?v=acfAf6eLbh8)


#### Note: This work can be used freely for academic research work and individual non-commercial projects, please do provide citation and/or deserved credits to this work. For Industrial and commercial use permission is required from the Author.
