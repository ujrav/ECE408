ECE 408 FA15 Final Project - Parallelizing Face Detection via Haar Classifiers using CUDA
=========================

Umberto Ravaioli and Sharon Tang

## Setting Up Our Project to Run on Windows Visual Studios (not tested for Linux/Unix)

###A CUDA-capable GPU

###Visual Studios Community 2013
[Download VS](https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx)

###CUDA Toolkit
[Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

[Read more from NVIDIA if you are having difficulties with running any CUDA project on Windows or Visual Studios](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html#introduction)

---

## Get our code from Github

Grab code from the master branch

~~~
$ git clone https://github.com/ujrav/ECE408.git ece408
~~~

Navigate to the Visual Studios Project
~~~
$ cd ../ece408/parallel/parallel
~~~

Find "parallel.vcxproj" and open in Visual Studios.

---

## Compiling the CUDA project and running the program

In the function main of kernel.cu, there is a line to change the image input.

~~~
image = readBMP("Images/<name>.bmp", width, height);
~~~

Our program is only designed to read BMP images. You can run the image tests already provided in the Images folder
or you can create your own. Directions on creating usable test images are found below. Please limit the image size
to no more than 1000x1000.

Hit R7 or Click on Build > Build Solution and wait for it to compile. (There will be some warnings about the RapidXML file).

Once it is done, hit Ctrl+F5 to run the program.

When completed, there should be two image outputs in the Image folder: "Images/output.bmp" and "Images/outputSerial.bmp".

---

## Creating usable images for the program.

Take any image you want of any format of your choosing. Limit the size to at most 1000x1000 resolution.

Convert the image to BMP (24-bit). If on Windows, open the image in Microsoft Paint. Click File > Save As > Select BMP and save it
into the Images folder.

Your image should be ready to be run by the program!
