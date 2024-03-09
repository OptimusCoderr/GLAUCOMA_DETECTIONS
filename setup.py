
from setuptools import find_packages,setup

setup(
    name='detect_glaucoma',
    version='0.0.1',
    author='OptimusCoderr',
    author_email='chukwuebukaanulunko@gmail.com',
    install_requires=["keras_cv","keras","tensorflow","opencv-python", "pandas","numpy","matplotlib","streamlit","Pillow"],
    packages=find_packages()
)