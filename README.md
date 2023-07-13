
# Federated Machine Learning with Flower: Development and Monitoring

## Workshop description
This repository contains all the code, documentation, and resources you will need to participate in our workshop on Federated Learning using the [Flower package](https://flower.dev/). Flower is a powerful and relatively easy-to-use open-source framework for Federated Learning, allowing developers to easily create and manage distributed machine learning models.

Federated Learning is a development in machine learning that allows models to be trained on decentralized data without the need for centralization. It preserves data privacy, reduces data transfer costs, and enables faster model training. This approach has applications in various industries, including healthcare, finance, and transportation, where data privacy and security are critical. 

The decentralized nature of federated learning brings new challanges with regards to deploying, packaging, monitoring and training these models. In this workshop we will be discussing these challenges, and we will show you how you could tackle them.

## Requirements
python, docker

## Part one: basic setup with limited monitoring
* Clone the repository
* run `make part_1` to run the first part of the demo
* The dockerized clients will start training on their local data and then send model coefficients to the dockerized flower server.
* The flower server aggregates these coefficients and shares them back with the clients.
* There is very limited monitoring and logging, only metrics logging to stdout on the client side
* hit `ctrl-c` to stop

## Part two: adding a monitoring dashboard
* run `make part_2` to run the second part of the demo
* The server now has access to the full training set and a test set to evaluate performance against
* A central model is fit on the full training set to compare performance and coefficients against
* The clients split their data into train and test data, and report metrics for both an local-only (edge) train model and the federated model
* Server side metrics get stored to a metrics.json file
* A dashboard is launched on [http://localhost:8050](http://localhost:8050) where training progress can be monitored
* hit `ctrl-c` to stop


## Dev
* Run `make install` to install python dependencies for linting, formatting and pre-commit hooks (ruff and black)
* Run a bare `make` to see other options (such as linting, formatting, etc)


## Credits
This workshop was set up by the [MLOps and Crafts meetup](https://www.meetup.com/nl-NL/mlops-and-crafts/), an initiative of [Thoughtworks Netherlands](www.thoughtworks.com), by [@atroyanovsky](https://github.com/atroyanovsky), [@saraperric85](https://github.com/saraperric85) and [@oegedijk](https://github.com/oegedijk).

The workshop code is partly based on the [following flwr sklearn example](https://flower.dev/docs/quickstart-scikitlearn.html).

Github: [github.com/mlops-and-crafts/federated-learning](https://github.com/mlops-and-crafts/federated-learning)
