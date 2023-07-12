
# Federated Machine Learning with Flower: Development and Monitoring

## Workshop description
This repository contains all the code, documentation, and resources you will need to participate in our workshop on Federated Learning using the [Flower package](https://flower.dev/). Flower is a powerful and relatively easy-to-use open-source framework for Federated Learning, allowing developers to easily create and manage distributed machine learning models.

Federated Learning is a development in machine learning that allows models to be trained on decentralized data without the need for centralization. It preserves data privacy, reduces data transfer costs, and enables faster model training. This approach has applications in various industries, including healthcare, finance, and transportation, where data privacy and security are critical. 

The decentralized nature of federated learning brings new challanges with regards to deploying, packaging, monitoring and training these models. In this workshop we will be discussing these challenges, and we will show you how you could tackle them.

## Requirements
Python 3.9, docker

## Usage
* Clone the repository
* run `make run` from the commandline to start the `docker-compose` containers with a server, a number of clients and a dashboard.
* The clients will start training on their local data and then send model coefficients to the server.
* The server aggregates these coefficients and shares them back with the clients.
* Metrics get logged in `metrics.json`, which can be tracked and displayed on the dashboard
* go to [http://localhost:8050](http://localhost:8050) to see the training progress
* hit `ctrl-c` to detach the logs and run `make stop` to bring down the containers

## Dev
* Run `make install` to install python dependencies for linting, formatting and pre-commit hooks (ruff and black)
* While developing you can run `make restart` to shut down and relaunch the containers
* Run a bare `make` to see other options (such as linting, formatting, etc)


## Credits
This workshop was set up by the [MLOps and Crafts meetup](https://www.meetup.com/nl-NL/mlops-and-crafts/), an initiative of [Thoughtworks Netherlands](www.thoughtworks.com), by [@atroyanovsky](https://github.com/atroyanovsky), [@saraperric85](https://github.com/saraperric85) and [@oegedijk](https://github.com/oegedijk).

The workshop code is partly based on the [following flwr sklearn example](https://flower.dev/docs/quickstart-scikitlearn.html).

Github: [github.com/mlops-and-crafts/federated-learning](https://github.com/mlops-and-crafts/federated-learning)
