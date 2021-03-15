from setuptools import setup, find_packages

setup(
    name='face_expression',
    version='1.0',
    author='Karim Iskakov',
    install_requires=['numpy'],
    packages=find_packages(),
    include_package_data=True,
)
