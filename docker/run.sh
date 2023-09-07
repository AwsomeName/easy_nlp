sudo docker run -it test:v1 /bin/bash
sudo docker run -it test:v2 /bin/bash
sudo docker run  --gpus all -it -p 11073:11073 -v /home/lc/models/:/root/data test:v2 /bin/bash