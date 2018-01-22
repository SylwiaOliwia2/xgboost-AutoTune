from distutils.core import setup

setup(
    name='xgboost_autotune',
    version='1.0',
    description='Xgboost_autotune is a Python module which auto-tunes bosting models.',
    author='Sylwia Mielnicka',
    author_email='hello@sylwiamielnicka.com',
    url='https://github.com/SylwiaOliwia2/xgboost-AutoTune',
    py_modules=['xgboost_autotune'],
    install_requires=['sklearn', 'numpy', 'copy'],
)
