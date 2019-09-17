#/bin/bash
python3 train_synthText.py --gt_path=/app/SynthText/gt.mat \
--synth_dir=/app/SynthText \
--label_size=96 \
--batch_size=16 \
--test_batch_size=16 \
--test_interval=40 \
--max_iter=50000 \
--lr=0.0001 \
--epochs=800 \
--test_iter=10

