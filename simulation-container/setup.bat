docker build -t basilisk-simulation .
docker run -it -v %CD%\simulations:/basilisk/simulations --name basilisk_container basilisk-simulation
