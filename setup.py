from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='skipi',
    description='Intuitive package to easily work with mathematical functions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=__import__('skipi').__version__,
    author='Alexander Book',
    author_email='alexander.book@frm2.tum.de',
    license='MIT',
    url='https://github.com/TUM-E21-ThinFilms/skipi',
    packages=find_packages(),
    keywords=['scientific', 'mathematical transforms'],
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'matplotlib'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
