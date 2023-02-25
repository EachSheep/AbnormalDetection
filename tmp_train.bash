log_label=1
for embedding_dim in 320 360 400 ; do
    for ffn_num_hiddens in 160 240 320 ; do
        for num_heads in 2 4 8 ; do
            for num_layers in 2 4 8 ; do
                for dropout in 0.5 ; do
                    for lr in 0.0002 ; do
                        for steps_per_epoch in 40 80 120 ; do
                            for batch_size in 64 128 256 512 ; do
                                python train.py -dataset_root=/home/hiyoungshen/Source/ICWS2023/AbnormalDetection/experiment/preprocess/ \
                                            -weight_name model.pkl \
                                            -file_name_abnormal feedback.csv \
                                            -file_name_normal normal.csv \
                                            --use_cache \
                                            -data_type pageuser \
                                            -max_seq_len 300 \
                                            -vocab_dict_path experiment/assets/page2idx.json \
                                            -vocab_size 10000 \
                                            -backbone transformer \
                                            -embedding_dim $embedding_dim \
                                            -ffn_num_hiddens $ffn_num_hiddens \
                                            -num_heads $num_heads \
                                            -num_layers $num_layers \
                                            -dropout $dropout \
                                            -criterion BCE \
                                            -lr $lr \
                                            -epochs 30 \
                                            -steps_per_epoch $steps_per_epoch \
                                            -batch_size $batch_size \
                                            -log_label $log_label \
                                            -train_ratio 0.8
                                log_label=$((log_label+1))
                            done
                        done
                    done
                done
            done
        done
    done
done
