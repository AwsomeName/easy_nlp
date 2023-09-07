# 安装docker
sudo apt update 
sudo apt install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common

## 添加docker的GPG密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - 
sudo apt-key fingerprint 0EBFCD88

## 添加docker的APT仓库
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

## 检查NVIDIA 仓库源是否添加
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) 
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - 
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update


## 安装docker ce和 NVIDIA docker 2
sudo apt-get update 
sudo apt-get install -y docker-ce docker-ce-cli containerd.io nvidia-docker2


## 启用docker服务，并设置自启
sudo systemctl enable docker 
sudo systemctl start docker

## 验证docker是否安装成功
docker --version


## 注册登录docker账号

## 安装测试镜像
# docker pull nvidia/cuda:11.4.0-base
docker pull nvidia/cuda:12.1.0-devel-ubuntu18.04
# 这个必须去docker去找


## 测试docker GPU支持
# docker run --gpus all nvidia/cuda:11.4.0-base nvidia-smi
docker run --gpus all nvidia/cuda:12.1.0-devel-ubuntu18.04 nvidia-smi
# docker pull nvidia/cuda:12.1.0-devel-ubuntu18.04


# torch调用