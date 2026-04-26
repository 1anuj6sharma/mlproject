# Using setup.py we make a machine learning project as the package 

from setuptools import setup, find_packages

from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    
    This function will return the list of requirements

    '''
    requirements=[]
    
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()  # when we go to the next line then here /n will be come so we will remove it by using strip function
        requirements=[req.replace("\n","") for req in requirements]
        
        
        if HYPEN_E_DOT in requirements:  # when we are installing the package then this line will be come in the requirements.txt file and we have to remove it because this is not a requirement for our project
            requirements.remove(HYPEN_E_DOT)
            
            
    return requirements       
    



setup(
    name='miproject',
    version='0.0.1',
    author='Anuj',
    author_email='anujpachauri441@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
