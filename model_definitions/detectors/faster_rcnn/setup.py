# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "roi_layers", "c_src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "roi_layers._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="faster_rcnn",
    version="0.1",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)

    #TODO add instructions to readme
"""
Correct Output:

/home/.../bin/python /.../pytorch.repmet/model_definitions/detectors/faster_rcnn/setup.py build develop
running build
running build_py
creating build
creating build/lib.linux-x86_64-3.6
creating build/lib.linux-x86_64-3.6/roi_layers
copying roi_layers/roi_align.py -> build/lib.linux-x86_64-3.6/roi_layers
copying roi_layers/roi_pool.py -> build/lib.linux-x86_64-3.6/roi_layers
copying roi_layers/nms.py -> build/lib.linux-x86_64-3.6/roi_layers
copying roi_layers/__init__.py -> build/lib.linux-x86_64-3.6/roi_layers
creating build/lib.linux-x86_64-3.6/rpn
copying rpn/generate_anchors.py -> build/lib.linux-x86_64-3.6/rpn
copying rpn/rpn_target.py -> build/lib.linux-x86_64-3.6/rpn
copying rpn/proposal_layer.py -> build/lib.linux-x86_64-3.6/rpn
copying rpn/rpn.py -> build/lib.linux-x86_64-3.6/rpn
copying rpn/__init__.py -> build/lib.linux-x86_64-3.6/rpn
running build_ext
building 'roi_layers._C' extension
creating build/temp.linux-x86_64-3.6
creating build/temp.linux-x86_64-3.6/.
creating build/temp.linux-x86_64-3.6/..
creating build/temp.linux-x86_64-3.6/...
creating build/temp.linux-x86_64-3.6/...
creating build/temp.linux-x86_64-3.6/.../pytorch.repmet
creating build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions
creating build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors
creating build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn
creating build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers
creating build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src
creating build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cpu
creating build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda
gcc -pthread -B /home/.../compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -DWITH_CUDA -I/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src -I/home/.../lib/python3.6/site-packages/torch/lib/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/TH -I/home/.../lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/home/.../include/python3.6m -c /.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/vision.cpp -o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/vision.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
gcc -pthread -B /home/.../compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -DWITH_CUDA -I/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src -I/home/.../lib/python3.6/site-packages/torch/lib/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/TH -I/home/.../lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/home/.../include/python3.6m -c /.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cpu/nms_cpu.cpp -o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cpu/nms_cpu.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
gcc -pthread -B /home/.../compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -DWITH_CUDA -I/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src -I/home/.../lib/python3.6/site-packages/torch/lib/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/TH -I/home/.../lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/home/.../include/python3.6m -c /.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cpu/ROIAlign_cpu.cpp -o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cpu/ROIAlign_cpu.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
/usr/local/cuda/bin/nvcc -DWITH_CUDA -I/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src -I/home/.../lib/python3.6/site-packages/torch/lib/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/TH -I/home/.../lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/home/.../include/python3.6m -c /.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda/ROIPool_cuda.cu -o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda/ROIPool_cuda.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
/usr/local/cuda/bin/nvcc -DWITH_CUDA -I/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src -I/home/.../lib/python3.6/site-packages/torch/lib/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/TH -I/home/.../lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/home/.../include/python3.6m -c /.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda/ROIAlign_cuda.cu -o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda/ROIAlign_cuda.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
/usr/local/cuda/bin/nvcc -DWITH_CUDA -I/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src -I/home/.../lib/python3.6/site-packages/torch/lib/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/home/.../lib/python3.6/site-packages/torch/lib/include/TH -I/home/.../lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/home/.../include/python3.6m -c /.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda/nms.cu -o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda/nms.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
g++ -pthread -shared -B /home/.../compiler_compat -L/home/.../lib -Wl,-rpath=/home/.../lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/vision.o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cpu/nms_cpu.o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cpu/ROIAlign_cpu.o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda/ROIPool_cuda.o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda/ROIAlign_cuda.o build/temp.linux-x86_64-3.6/.../pytorch.repmet/model_definitions/detectors/faster_rcnn/roi_layers/c_src/cuda/nms.o -L/usr/local/cuda/lib64 -lcudart -o build/lib.linux-x86_64-3.6/roi_layers/_C.cpython-36m-x86_64-linux-gnu.so
running develop
running egg_info
creating faster_rcnn.egg-info
writing faster_rcnn.egg-info/PKG-INFO
writing dependency_links to faster_rcnn.egg-info/dependency_links.txt
writing top-level names to faster_rcnn.egg-info/top_level.txt
writing manifest file 'faster_rcnn.egg-info/SOURCES.txt'
reading manifest file 'faster_rcnn.egg-info/SOURCES.txt'
writing manifest file 'faster_rcnn.egg-info/SOURCES.txt'
running build_ext
copying build/lib.linux-x86_64-3.6/roi_layers/_C.cpython-36m-x86_64-linux-gnu.so -> roi_layers
Creating /home/.../lib/python3.6/site-packages/faster-rcnn.egg-link (link to .)
faster-rcnn 0.1 is already the active version in easy-install.pth

Installed /.../pytorch.repmet/model_definitions/detectors/faster_rcnn
Processing dependencies for faster-rcnn==0.1
Finished processing dependencies for faster-rcnn==0.1

Process finished with exit code 0
"""