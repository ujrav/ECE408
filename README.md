ECE 408 FA15 Final Project - Parallelizing Face Detection via Haar Classifiers using CUDA
=========================

## Umberto Ravaioli and Sharon Tang

Welcome to our ECE 408 Applied Parallel Programming final project repository.

The goal of this project was to speed up the process of face detection using CUDA and the parallelization techniques
we discussed in our classes. We first implemented our own serial version of the algorithm as a benchmark to test our
parallelized program against. We then proceeded to apply various methods of parallelization to the Haar Cascading
process with the goal of achieving a significant speedup from the serial version.

The master branch of our repository contains our most recent code. Feel free to grab it and play around with it yourself.


## What You Need To Run Our Project on Windows 7, 8, or 10 (not tested on Linux/Unix)

###A CUDA-capable GPU

###Visual Studios Community 2013
[Download VS](https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx)

###CUDA Toolkit
[Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

---

## Setting up CUDA with Visual Studios

Refer to the following:
[From NVIDIA](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html#introduction)

[More directions on setting up CUDA with Visual Studios](http://cuda-programming.blogspot.com/2013/01/installation-process-how-to-install.html)

---

## Get our code from Github

Grab code from the master branch

~~~
$ git clone https://github.com/ujrav/ECE408.git ece408
~~~

Navigate to the Visual Studios Project file.

~~~
$ cd ../ece408/parallel/parallel
~~~

Find "parallel.vcxproj" and open it in Visual Studios.

---

## Compiling the CUDA project and running the program

In the function main of "kernel.cu", there is a line to change the image input.

~~~
image = readBMP("Images/<name>.bmp", width, height);
~~~

Our program is only capable of reading BMP images at the moment. You can run the image tests already provided in
the Images folder or you can create your own. Directions on creating usable test images are found below. Please
limit the image size to no more than 1000x1000.

Hit R7 or Click on Build > Build Solution and wait for it to compile. (There will be some warnings about the RapidXML file).

Once it is done, hit Ctrl+F5 to run the program.

When completed, there should be two image outputs in the Image folder: "Images/output.bmp" and "Images/outputSerial.bmp".
Open them and check to make sure green boxes have been drawn around the human faces in the image.

---

## Creating usable images for the program

Take any image you want of any format of your choosing. Limit the size to at most 1000x1000 resolution.

Convert the image to BMP (24-bit). If on Windows, open the image in Microsoft Paint. Click File > Save As > Select BMP and save it
into the Images folder.

Your image should be ready to be run by the program!
