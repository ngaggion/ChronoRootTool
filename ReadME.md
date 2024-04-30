# ChronoRoot 2.0

ChronoRoot 2.0 is a comprehensive tool for plant root phenotyping, comprising two distinct stages: segmentation using the state-of-the-art method [nnUNet](https://github.com/MIC-DKFZ/nnUNet), and a user interface for conducting the phenotyping procedure and generating reports. It offers flexibility for local installation or can be utilized through a Docker image.

## Local Installation of ChronoRootInterface

To install the ChronoRoot interface locally using Anaconda and pip packages, follow these steps:

```bash
conda create -y -n ChronoRootInterface python=3.8
conda activate ChronoRootInterface
conda install -c "conda-forge/label/cf202003" graph-tool=2.29 pyqt=5.9.2
conda install numpy scikit-image pandas seaborn
conda install -c conda-forge pyzbar
pip install opencv-python
```

Alternatively, you can use the provided environment.yml file in the repository.

### Usage

Activate the ChronoRootInterface environment:

```bash
conda activate ChronoRootInterface
```

Then, run the interface:

```bash
python run.py
```

## Local Installation of nnUNet

For local installation of nnUNet, minor modifications were made to allow soft dense segmentation outputs for temporal post-processing. Follow the instructions in the repository [github.com/ngaggion/nnUNet](https://github.com/ngaggion/nnUNet) to set it up.

### Usage

Before running the segmentation pipeline, download nnUNet models and data [available here](https://huggingface.co/datasets/ngaggion/ChronoRoot_nnUNet) inside the "Segmentation" folder. You'll need [git-lfs](https://git-lfs.com/) to clone the dataset from HuggingFace. Install git-lfs, then clone the dataset:

```bash
cd Segmentation
git lfs install
git clone https://huggingface.co/datasets/ngaggion/ChronoRoot_nnUNet
```

Modify the bash file with paths to your videos individually. An example is provided:

```bash
cd Segmentation
conda activate nnUNet
./segment.sh
```

## Combined ChronoRoot Docker Image

This Docker image provides an environment for running ChronoRoot completely, both for segmentation and with the user interface, along with GPU support and X11 forwarding.

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
MOUNT="YOUR_LOCAL_INFORMATION_PATH"

docker run -it --gpus all \
    -v $MOUNT:/DATA/ \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    chronoroot_full:latest
```

It's recommended to always pull from the repo when starting the docker.

```bash
git pull
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