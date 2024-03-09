
from setuptools import find_packages,setup

setup(
    name='mcqgenerator',
    version='0.0.1',
    author='OptimusCoderr',
    author_email='chukwuebukaanulunko@gmail.com',
    install_requires=["keras_cv", "keras","tensorflow","opencv-python", "pandas","numpy","tqdm", "joblib", "matplotlib","streamlit"],
    packages=find_packages()
)