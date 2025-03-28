# outperforming-cublas

copying https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog and https://github.com/bertmaher/simplegemm to learn more!

personal note: build command is:
```
CUDNN_LIBRARY_PATH=/usr/lib64/libcudnn.so.8.9.7 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=9.0a USE_GOLD_LINKER=1 with-proxy python setup.py develop
```
