
= GTE001: GT_CUDA_SRC env variable not set.

This error occurs when the `gpu` feature is enabled and `gt-cuda` can't locate the install path of cuda. To fix, export your the environment variable `GT_CUDA_SRC` in your `~/bashrc`. For an in-depth guide on Cuda setup for gT, see `cuda_setup.pdf` in the docs.

= GTE002: Cuda Initialization Failed.

The cuda driver failed to initialize on `cuInit()`. This occurs when your GPU does not have a valid driver. 

= GTE003: Failed to get cuda device count during initialization.

The cuda driver successfully initialized, but the subsequent call to `cuDeviceGetCount` failed. Not sure why this would happen. Try updating drivers.

= GTE004: Failed to get cuda device during initialization.

The cuda driver successfully initialized, but the subsequent call to `cuDeviceGet` failed. Not sure why this would happen. Try updating drivers.

= GTE005: Failed to get cuda device attribute.

The cuda driver successfully initialized, but the subsequent call to `cuDeviceGetAttribute` failed. Try updating drivers.

= GTE006: Gpu Invalid Ordinal

This error occurs when you call Gpu::new() for a device ordinal that does not exist. Try using Gpu::strongest() instead. 

= GTE007: Failed to create Gpu Context. 

This error occurs when the call to `cuCtxCreate` fails when creating a new Gpu context through `Gpu::new`. 

= GTE008: Failed to bind to thread.

This error occurs when the call to `cuCtxSetCurrent` fails. This binds the `Gpu` object to the calling thread. 

= GTE009: No Devices Found.

You tried to create a `Gpu`, but no devices were found.

= GTE010: Failed to compile cuda program.

