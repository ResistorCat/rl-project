# Use the official Node.js LTS (Long Term Support) image.
# https://hub.docker.com/_/node
FROM node:lts

# Set a working directory in the container.
WORKDIR /usr/src/app/pokemon-showdown

#  Install git and clone the Pokemon Showdown repository.
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/smogon/pokemon-showdown.git /usr/src/app/pokemon-showdown && \
    cd /usr/src/app/pokemon-showdown && \
    node build && \
    cp config/config-example.js config/config.js && \
    rm -rf /var/lib/apt/lists/*

