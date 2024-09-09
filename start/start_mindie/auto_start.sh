docker run -it -d \
--name MindIE-test1 \
--ipc=host \
--net=host \
--restart=always \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--env-file=./mindie-env.list \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
-v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/vnpu.cfg:/etc/vnpu.cfg \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /home/aicc:/home/aicc \
-v /data/nv0/tlf/models/:/modelfiles/  \
-v /data/nv0/tlf/models/conf/:/usr/local/Ascend/mindie/latest/mindie-service/conf \
swr.cn-central-221.ovaijisuan.com/wh-aicc-fae/mindie:37131-ascend_24.1.rc2-cann_8.0.rc2-py_3.10-ubuntu_22.04-aarch64-mindie_1.0.RC2.01 \
/usr/local/Ascend/mindie/latest/mindie-service/bin/mindieservice_daemon
# /bin/bash
# printenv 