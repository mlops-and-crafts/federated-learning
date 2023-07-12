.PHONY: help run server clients stop restart log build lint

# Display this help
help:
	@echo "Available targets:"
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^# (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 1, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 2, RLENGTH); \
			printf "\033[36m%-20s\033[0m %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

# install dev dependencies and set pre-commit hook
install:
	pip install ruff black pre-commit
	pre-commit install

# Run the entire application locally, dashboard on localhost:8050
run:
	docker-compose up -d && docker-compose logs -f -t

# Start only the server container
server:
	docker-compose up server -d

# Start only the client containers
clients:
	docker-compose up client -d

# start only the dashboard container
dashboard:
	docker-compose up dashboard -d

# Stop and remove all containers
stop:
	docker-compose down

# Restart the application
restart:
	docker-compose down
	docker-compose up -d && docker-compose logs -f -t

# Show the logs of all containers
log:
	docker-compose logs -f -t

# rebuild the containers without cache
build:
	docker-compose build --no-cache

# format the python code using black
format:
	black federated-learning-workshop

# Lint the python code using ruff and Dockerfiles using hadolint
lint:
	ruff federated-learning-workshop
	docker run --rm -i hadolint/hadolint < federated-learning-workshop/Dockerfile.client
	docker run --rm -i hadolint/hadolint < federated-learning-workshop/Dockerfile.server
	docker run --rm -i hadolint/hadolint < federated-learning-workshop/Dockerfile.dashboard