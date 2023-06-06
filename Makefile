.PHONY: server client

run: 
	docker-compose up -d 
	
server:
	docker-compose up server -d 

clients:
	docker-compose up client -d

stop:
	docker-compose down


