import os
import time
import docker
import logging
from docker.errors import ContainerError, APIError, BuildError
from docker.models.containers import Container


class PokemonShowdownServer:
    def __init__(self, port: int = 8000):
        self.port = port
        self.client = docker.from_env()
        self.logger = logging.getLogger("PokemonShowdownServer")

    def start(self) -> bool:
        self.logger.info(f"Starting Pokemon Showdown server on port {self.port}...")
        self.logger.info(
            "If you are running this for the first time, it may take ~2 minutes to build the Docker image."
        )
        try:
            start_time = time.time()
            self.client.images.build(
                path=os.path.join(os.path.dirname(__file__), "docker"),
                dockerfile="Dockerfile",
                tag="pokemon-showdown",
                rm=True,
            )
            self.client.containers.run(
                "pokemon-showdown",
                detach=True,
                ports={f"{self.port}/tcp": self.port},
                command="node pokemon-showdown start --no-security",
            )
            self.logger.info(
                f"Pokemon Showdown server started on port {self.port} (took {time.time() - start_time:.2f} seconds)."
            )
            return True
        except BuildError as e:
            self.logger.error(f"Error building Docker image: {e}")
            if e.build_log:
                for line in e.build_log:
                    self.logger.error(line.get("stream", "").strip())
        except ContainerError as e:
            self.logger.error(f"Error starting Pokemon Showdown server: {e}")
        except APIError as e:
            self.logger.error(f"Docker API error: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
        return False

    def stop(self) -> bool:
        self.logger.info("Stopping Pokemon Showdown server...")
        try:
            stop_time = time.time()
            containers: list[Container] = self.client.containers.list(
                filters={"ancestor": "pokemon-showdown"}
            )
            for container in containers:
                container.stop()
                container.remove()
            self.logger.info(
                f"Pokemon Showdown server stopped. (took {time.time() - stop_time:.2f} seconds)."
            )
            return True
        except ContainerError as e:
            self.logger.error(f"Error stopping Pokemon Showdown server: {e}")
        except APIError as e:
            self.logger.error(f"Docker API error: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
        return False

    def restart(self):
        self.stop()
        self.start()

    def is_running(self) -> bool:
        try:
            containers: list[Container] = self.client.containers.list(
                filters={"ancestor": "pokemon-showdown"}
            )
            return len(containers) > 0
        except APIError as e:
            self.logger.error(f"Docker API error: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
        return False


if __name__ == "__main__":
    # Test the server functionality
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )
    server = PokemonShowdownServer(port=8000)
    if server.start():
        input("Press Enter to stop the server...")
        server.stop()
    else:
        server.logger.error("Failed to start the server.")
        server.restart()
