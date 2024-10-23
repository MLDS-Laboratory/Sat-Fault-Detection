docker build -t basilisk-sim .
docker run -it -v %CD%\simulations:/basilisk/simulations --name basilisk_container basilisk-sim
