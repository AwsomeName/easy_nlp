sudo docker run --name ubunut-nginx -it -d \
        -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
        --ipc=host --net=host \
        ubuntu:22.04 /bin/bash -c "sleep infinity"
        # -v /usr/local/sbin:/usr/local/sbin \
        # -v /usr/local/dcmi:/usr/local/dcmi \
