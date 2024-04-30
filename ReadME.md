# ChronoRoot 2.0

ChronoRoot2.0 consists of two different stages: the segmentation method, which is performed using the state-of-the-art segmentation method [nnUNet](https://github.com/MIC-DKFZ/nnUNet), and a user interface to perform the plant root phenotyping procedure and report generation.

Thus, it consists of two different conda environments which can be installed separately, or used with the docker image provided below.

## Local Installation of ChronoRootInterface

With anaconda and pip packages:

```bash
conda create -y -n ChronoRootInterface python=3.8
conda activate ChronoRootInterface
conda install -c "conda-forge/label/cf202003" graph-tool=2.29 pyqt=5.9.2
conda install numpy scikit-image pandas seaborn
conda install -c conda-forge pyzbar
pip install opencv-python
```

### Usage

```bash
conda activate ChronoRootInterface
python run.py
```

Or using the environment.yml file present in this repo.

## Local Installation of nnUNet

As minor modifications where included in the nnUNet repo, to allow soft dense segmentation outputs which are used in the temporal post-processing steps, please follow the instructions from the following repo: [github.com/ngaggion/nnUNet](https://github.com/ngaggion/nnUNet)

### Usage

The segmentation pipeline requires to download nnUNet models and data first, [available here](https://huggingface.co/datasets/ngaggion/ChronoRoot_nnUNet) inside "Segmentation" folder. Please note that we also provide the segmentation dataset, which can be enhanced by adding extra annotations to fineture your models easily following the nnUNet instructions.

You will need [git-lfs](https://git-lfs.com/) to clone the dataset from HuggingFace, follow the instructions to install then run:

```bash
cd Segmentation
git lfs install
git clone https://huggingface.co/datasets/ngaggion/ChronoRoot_nnUNet
```

To run the segmentation scripts, modify the bash file with the paths to your videos, individually. An example is provided.

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