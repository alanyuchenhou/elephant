from setuptools import setup, find_packages

setup(name='elephant',
      version='0.1',
      description='graph mining with artificial neural nets',
      url='https://github.com/yuchenhou/elephant',
      author='Yuchen Hou',
      author_email='houych@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['requests', 'numpy', 'scipy', 'scikit-learn', 'tensorflow', 'pandas', ],
      zip_safe=False)
