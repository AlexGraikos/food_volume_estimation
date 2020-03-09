import logging
from setuptools import setup, find_packages

# Load requirements from file
try:
    with open('requirements.txt', 'r') as req_file:
        install_reqs = req_file.read()
except Exception:
    logging.warning('[!] Failed at loading requirements file.')

print(find_packages())

setup(
    name='food-volume-estimation',
    version='0.3',
    description='Estimate food volume from input image.',
    url='https://github.com/AlexGraikos/food_volume_estimation',
    author='Graikos Alexandros',
    author_email='graikosal@gmail.com',
    install_requires=install_reqs,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Educaction',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='food volume estimation tensorflow keras',
    packages=find_packages()
)

