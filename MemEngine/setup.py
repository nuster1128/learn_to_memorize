from setuptools import setup, find_packages

with open("requirements.txt",encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='memengine',
    version='1.0.2',
    author='Zeyu Zhang',
    author_email='zeyuzhang@ruc.edu.cn',
    description='A Comprehensive Library for Memory of LLM-based Agents.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nuster1128/MemEngine',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=requirements
)