"""
Docker utility functions for the Pokémon RL project.
"""

import docker


def check_docker_availability():
    """
    Check if Docker is available and running.

    Returns:
        bool: True if Docker is available and running, False otherwise.
    """
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False
