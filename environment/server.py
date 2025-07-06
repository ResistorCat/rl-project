import os
import time
import socket
import docker
import logging
from docker.errors import ContainerError, APIError, BuildError
from docker.models.containers import Container


class PokemonShowdownServer:
    def __init__(self, port: int = 8000):
        self.port = port
        self.client = docker.from_env()
        self.logger = logging.getLogger("PokemonShowdownServer")

    def test_connectivity(self) -> bool:
        """Test if the server is accessible on the configured port."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                result = sock.connect_ex(('localhost', self.port))
                if result == 0:
                    self.logger.info(f"✅ Server is accessible on port {self.port}")
                    return True
                else:
                    self.logger.warning(f"❌ Server is not accessible on port {self.port}")
                    return False
        except Exception as e:
            self.logger.error(f"Error testing connectivity: {e}")
            return False

    def start(self) -> bool:
        self.logger.info(f"Starting Pokemon Showdown server on port {self.port}...")
        
        # First, check if there are any existing containers and clean them up
        if self.is_running():
            self.logger.info("Found existing Pokemon Showdown containers. Stopping them first...")
            if not self.stop():
                self.logger.warning("Could not stop existing containers, but continuing...")
        
        self.logger.info(
            "If you are running this for the first time, it may take ~2 minutes to build the Docker image."
        )
        try:
            start_time = time.time()
            
            # Build the image
            self.logger.info("Building Docker image...")
            build_result = self.client.images.build(
                path=os.path.join(os.path.dirname(__file__), "docker"),
                dockerfile="Dockerfile",
                tag="pokemon-showdown",
                rm=True,
            )
            # build_result is a tuple (image, build_logs)
            if isinstance(build_result, tuple):
                image, build_logs = build_result
                if image and hasattr(image, 'id') and image.id:
                    image_id = image.id[:12] if len(image.id) >= 12 else image.id
                    self.logger.info(f"Docker image built successfully: {image_id}")
                else:
                    self.logger.info("Docker image built successfully")
            else:
                self.logger.info("Docker image built successfully")
            
            # Start the container
            self.logger.info("Starting container...")
            container = self.client.containers.run(
                "pokemon-showdown",
                detach=True,
                ports={"8000/tcp": self.port},  # Map internal port 8000 to host port
                name=f"pokemon-showdown-{self.port}",  # Give it a consistent name
                remove=True,  # Remove container when it stops
            )
            
            # Wait a bit for the server to start
            self.logger.info("Waiting for server to initialize...")
            time.sleep(3)
            
            # Check if container is still running
            container.reload()
            if container.status != 'running':
                self.logger.error(f"Container failed to start. Status: {container.status}")
                logs = container.logs().decode('utf-8')
                self.logger.error(f"Container logs: {logs}")
                return False
            
            container_id = container.id[:12] if container.id else "unknown"
            self.logger.info(
                f"Pokemon Showdown server started successfully on port {self.port} "
                f"(took {time.time() - start_time:.2f} seconds). "
                f"Container ID: {container_id}"
            )
            
            # Test connectivity
            self.logger.info("Testing server connectivity...")
            time.sleep(2)  # Wait a bit more for server to be ready
            if self.test_connectivity():
                self.logger.info("✅ Server is ready and accessible!")
                return True
            else:
                self.logger.warning("⚠️ Server started but is not accessible. Checking logs...")
                logs = container.logs().decode('utf-8')
                self.logger.info(f"Container logs: {logs}")
                return True  # Consider it successful even if connectivity test fails
            
        except BuildError as e:
            self.logger.error(f"Error building Docker image: {e}")
            if hasattr(e, 'build_log') and e.build_log:
                for line in e.build_log:
                    if 'stream' in line:
                        self.logger.error(line['stream'].strip())
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
            
            # Find containers by image name
            containers: list[Container] = self.client.containers.list(
                filters={"ancestor": "pokemon-showdown"}, all=True
            )
            
            # Also try to find by name pattern
            named_containers = self.client.containers.list(
                filters={"name": f"pokemon-showdown-{self.port}"}, all=True
            )
            
            # Combine both lists
            all_containers = containers + named_containers
            # Remove duplicates
            unique_containers = list({c.id: c for c in all_containers}.values())
            
            if not unique_containers:
                self.logger.info("No Pokemon Showdown containers found to stop.")
                return True
            
            for container in unique_containers:
                try:
                    container_name = container.name if hasattr(container, 'name') else container.id[:12]
                    self.logger.info(f"Stopping container: {container_name}")
                    
                    if container.status == 'running':
                        container.stop(timeout=10)
                    container.remove(force=True)
                    self.logger.info(f"Container {container_name} stopped and removed")
                except Exception as e:
                    self.logger.warning(f"Error stopping container {container.id[:12]}: {e}")
            
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

    def restart(self) -> bool:
        self.logger.info("Restarting Pokemon Showdown server...")
        if not self.stop():
            self.logger.error("Failed to stop the server. Cannot restart.")
            return False
        if not self.start():
            self.logger.error("Failed to start the server after restart.")
            return False
        self.logger.info("Pokemon Showdown server restarted successfully.")
        return True

    def is_running(self) -> bool:
        try:
            # Check for containers by image
            containers: list[Container] = self.client.containers.list(
                filters={"ancestor": "pokemon-showdown"}
            )
            
            # Check for containers by name pattern
            named_containers = self.client.containers.list(
                filters={"name": f"pokemon-showdown-{self.port}"}
            )
            
            # Combine and check if any are running
            all_containers = containers + named_containers
            running_containers = [c for c in all_containers if c.status == 'running']
            
            if running_containers:
                self.logger.debug(f"Found {len(running_containers)} running Pokemon Showdown containers")
                return True
            
            return False
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
