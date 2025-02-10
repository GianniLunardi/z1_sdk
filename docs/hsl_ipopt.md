# HSL (Harwell Subroutines Library)

HSL provides a number of linear solvers that can be used in IPOPT to have faster computations (usually MA57 is sufficient for medium problem).

## Preparation
1. go to http://hsl.rl.ac.uk/ipopt
2. select the relevant source link, depending on the license needed; you can download either Coin-HSL Archive code or the Coin-HSL Full code
3. follow the instructions on the website, read the license, and submit the registration form
4. wait for an email containing a download link (this should take no more than one working day).
5. select the product and follow the intructions of [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL) for the installation of the subroutines
6. at the end of the installation, create a symbolic link to the `libcoinhsl.so`, since IPOPT will search for `libhsl.so` as deafult
```
cd <install-dir>/lib
sudo ln -s libcoinhsl.so libhsl.so
```
