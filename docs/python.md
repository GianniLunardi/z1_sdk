# Python scripts

## Environment
The environment can be built using `conda`:
```
conda env create -n ENVNAME --file environment.yml
conda activate ENVNAME
```
Then the following `pip` dependencies must be installed:
- **torch** (CPU version)
    ```
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    ```
- **acados** &rarr; follow the instructions on the [documentation](https://docs.acados.org/installation/index.html) site.
- **l4casadi** (compatibility ensured with version 1.4.1)
    ```
    pip install l4casadi==1.4.1 --no-build-isolation
    ```

## Examples
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