all:
	docker-compose -f docker-compose.yml  up -d --build

exec:
	docker exec -it python /bin/bash

down:
	docker-compose -f docker-compose.yml  down
	
clean: down
	yes | docker system prune -a

