

https://dev.to/ajeetraina/the-ollama-docker-compose-setup-with-webui-and-remote-access-via-cloudflare-1ion

docker compose up -d
docker compose — dry-run up -d 


Prerequisites
Before starting to set up the different components of our tutorial, make sure your system has the following:

Docker & Docker-Compose — Ensure Docker and Docker-Compose are installed on your system.
Milvus Standalone — For our purposes, we’ll use Milvus Standalone, which is easy to manage via Docker Compose; check out how to install it in our documentation
Ollama — Install Ollama on your system; visit their website for the latest installation guide.





https://abhiyantimilsina.medium.com/a-comparative-analysis-of-pdf-extraction-libraries-choosing-the-fastest-solution-3b6bd8588498


docker build . -t chat-app
docker run -d -p 8000:8000 --name chat-app