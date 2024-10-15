docker build -t basilisk-simulation .
docker run -it -v ${PWD}:/basilisk/simulations/ --name basilisk_container basilisk-simulation
