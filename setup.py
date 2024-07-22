from setuptools import find_packages, setup
import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob("*.cpp") + glob.glob("*.cu")


setup(
    name="mu2dm",
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cu_kernels",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
