docker build -t basilisk-simulation .
docker run -it -v ${PWD}\simulations\:/basilisk/simulations/ --name basilisk_container basilisk-simulation