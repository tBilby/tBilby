from setuptools import setup, find_packages

def get_requirements(kind=None):
    if kind is None:
        fname = "requirements.txt"
    else:
        fname = f"{kind}_requirements.txt"
    with open(fname, "r") as ff:
        requirements = ff.readlines()
    return requirements

setup(
    name='tbilby',
    description="A transdimensional Bayesian inference library utilizing Bilby.",
    version='0.0',     
    license="MIT",
        
    packages=find_packages(),
    
    install_requires=get_requirements(),
   )