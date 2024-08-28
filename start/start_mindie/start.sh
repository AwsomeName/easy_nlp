docker run -it -d --ipc=host --net=host\
        --device=/dev/davinci6  \
        --device=/dev/davinci7  \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -v /usr/local/sbin/:/usr/local/sbin/ \
        -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
        -v /var/log/npu/slog/:/var/log/npu/slog \
        -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump \
        -v /var/log/npu/:/usr/slog \
        -v /models/:/modelfiles/  \
        -v /models/mindie/mindie/:/installs/  \
        -p 3335:3335  \
        --name mindie-rc01-45-3335  mindie:rc1.0  /usr/bin/bash -c "sleep infinity"
