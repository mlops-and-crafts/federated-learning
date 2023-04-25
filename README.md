
# Workshop Name 
### Presentation: [Presentation_name](workshop/presentation_template.pptx)

## Workshop description
Describe why your topic is important and what you want to share with your audience

## Requirements
Indicate Python version and any other required tools

Add requirements.txt, conda.yml, pyproject.toml, Docker image or Binder/Google Collab link

## Usage
* Clone the repository
* Start { TOOL } and navigate to the workshop folder

start docker
colima start --memory 4 ## increase colima vm memory up to 4 GB
in client directory run by specifying ARG1= int from 0 to 55500:
docker build --build-arg ARG1=10 -t client .
in server directory run:
docker build -t sever . 
in project directory run:
docker compose up



## Video record
Re-watch [this video](link)

## Credits
This workshop was set up by @mlops-and-crafts and @atroyanovsky TODO ADD SARA
