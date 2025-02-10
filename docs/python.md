# Python scripts
Both simulations and real experiments can be launched from Python scripts (instead of C++ programs). The scripts can be found in the `z1_dk/example_py` folder.
The interface is designed in the `arm_python_interface.cpp` file. Users can change directly this source file to expose more methods and/or properties, following the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/). The `unitree_arm_interface.pyi` is simply a python interface for intelligent hints, which cannot be obtained from the compiled dynamic library `.so`.

The following `PYTHONPATH` must be added in the `.bashrc`:
```
export PYTHONPATH=$HOME/unitree_ws/src/z1_sdk/example_py:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/unitree_ws/src/z1_sdk/lib/:$LD_LIBRARY_PATH
```
Also, for visualization purposes, the following `ROS_PACKAGE_PATH` must be inserted:
```
export ROS_PACKAGE_PATH=$HOME/env_z1/lib/python3.9/site-packages/cmeel.prefix/share/example-robot-data:$ROS_PACKAGE_PATH
```
Some examples need python packages that were not listed in the Unitree requirements. In addition, `Python>=3.9` is needed. For full compatibility, the packages in `requirements.txt` must be installed (we suggest to build a virtual python environment):
```
pip install -r requirements.txt
``` 

## Common issues 
The installation of the requirements can be troublesome. In particular, we have faced issues with the following packages:
- `torch` -> we suggest the installation of the CPU version, since its needed only for inference using small MLP
    ```
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```
- `l4casadi` -> needs the `no-build-isolation` flag
    ```
    pip install l4casadi --no-build-isolation
    ```
Remember to upgrate pip to a version >=24.3.1

## Python bindings
If the user wants to expose more features in the python interface, then he must add the following lines in the `CMakeLists.txt`:
```
set(pybind11_DIR <dir-to-venv>/lib/python3.9/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)
```
where `<dir-to-venv>` is the path that points to the python virtual environment. During the building of the project, the compiler might not find `Eigen`. In most of the cases, this is due to an unproper linking that can be solved with the following:
```
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
```