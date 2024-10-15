docker start basilisk_container
docker exec -it basilisk_container bash -i -c "source .venv/bin/activate && exec bash"
