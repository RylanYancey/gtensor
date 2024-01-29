
#set heading(numbering: "1.1")

#show link: underline

  Instructions Outline:

  + Be on Linux or WSL.
  + Install the cuda toolkit and have a cuda-capable GPU.
  + Locate the cuda toolkit path.
  + Export `GT_CUDA_SRC` in your `~/.bashrc`.

= Installing Linux / WSL.

To keep the process of getting gpus working as simple as possible, I have decided to limit the feature to Linux only. If you don't have or don't want to install native Linux, you can opt to use the Windows Subsystem for Linux instead. WSL is terminal-only and runs inside of a Hypervisor (a type of hardware VM) within Windows, with performance similar to Native Linux. 

WSL2 Installation guide: #link("https://learn.microsoft.com/en-us/windows/wsl/install")

It is also a good idea to update your GPU drivers at this step.

= Installing the Cuda Toolkit

The gT recommended way to install the Cuda Toolkit is through the Spack Package Manager. Spack is a tool for installing and managing HPC (high-performance-computing) packages like Cuda or BLAS. Follow the guide below to install spack on your Linux machine or WSL instance. 

== Installing Spack

Spack Installation guide: #link("https://spack-tutorial.readthedocs.io/en/latest/tutorial_basics.html")

`spack install cuda@11.8.0`.

Now that spack is installed, you can run the above command to install the gT supported cuda version 11.8.0. Be forewarned this will take a while, as Spack will install all of cudas' dependencies locally, even if global versions are installed. This will take up more space on your drive, but it will make our cuda installation more stable and less likely to break in the future. 

= Locate the Cuda Toolkit Path

`spack location -i cuda`

Using the above command will return the path to the spack installation of cuda. 

= Export the GT_CUDA_SRC Environment Variable.

`nano ~/.bashrc`

Use the above command to edit your `bashrc` file. This file is ran whenever a new terminal instance is created.

`export GT_CUDA_SRC=$(spack location -i cuda)`

Add this line to any location in your `bashrc`. gTensor will use the `GT_CUDA_SRC` variable to find the cuda installation.

If you install cuda with `Environment Modules` or manually, just set `GT_CUDA_SRC` to the path you need.

