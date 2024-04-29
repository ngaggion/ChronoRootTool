# ChronoRoot 2.0

## ChronoRoot Docker Image

This Docker image provides an environment for running ChronoRoot, along with GPU support and X11 forwarding.

### Usage

Download the Docker image from the Docker Hub repository by running:

```bash
docker pull ngaggion/chronoroot_full:latest
```

If you are planning to use the user interface, execute the following commands to allow Docker to access the local X server:

```bash
xhost +local:docker
```

Then, run the Docker container with the following command:

```bash
docker run -it --gpus all \
    -v /media/ngaggion/DATA/Raices:/DATA/Raices \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    chronoroot_full:latest
```

After using the container, it's recommended to restrict access to the X server with the following command:

```bash
xhost -local:docker
```

### Docker Usage Notes

To enable GPU support within the Docker container, it's required to install the nvidia-docker2 package.

For Ubuntu-based distributions please follow these steps:

1. **Add the GPG key:**

    ```bash
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    ```

2. **Add the repository:**

    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/nvidia-docker/$distribution/$(arch)/" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```

3. **Update your package list:**

    ```bash
    sudo apt-get update
    ```

4. **Install NVIDIA Docker2:**

    ```bash
    sudo apt-get install -y nvidia-docker2
    ```