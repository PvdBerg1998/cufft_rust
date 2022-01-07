# cufft_rust

This is a safe Rust wrapper around CUDA FFT (`cuFFT`).

It only supports a subset of the API which I need for private projects. For now this only includes the real-to-complex forward transform.

The CUDA toolkit is not bundled and has to be installed manually. See [cuFFT](https://developer.nvidia.com/cufft).