from setuptools import setup, find_packages

package_name = 'lanenet'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    py_modules=[],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
        'tqdm'
    ],
    author='Andreas Klintberg',
    maintainer='Andreas Klintberg',
    description='Lanenet implementation in PyTorch',
    license='Apache License, Version 2.0',
    test_suite='pytest'
)
