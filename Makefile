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

# install dev dependencies and install pre-commit hooks
install:
	pip install ruff black pre-commit
	pre-commit install

# run the first part of the workshop locally
part_1:
	docker-compose -f part1/docker-compose.yml up

# run the second part of the workshop locally
part_2:
	docker-compose -f part2/docker-compose.yml up

# format the python code using black
format:
	black part1
	black part2

# Lint the python code using ruff and Dockerfiles using hadolint
lint:
	ruff part1
	ruff part2
	docker run --rm -i hadolint/hadolint < part1/Dockerfile.client
	docker run --rm -i hadolint/hadolint < part1/Dockerfile.server
	docker run --rm -i hadolint/hadolint < part2/Dockerfile.client
	docker run --rm -i hadolint/hadolint < part2/Dockerfile.server
	docker run --rm -i hadolint/hadolint < part2/Dockerfile.dashboard