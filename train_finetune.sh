#/bin/bash
python3 train_finetune.py --gt_path=/app/SynthText/gt.mat \
--synth_dir=/app/SynthText \
--ic13_root=/app/icdar2013 \
--label_size=96 \
--batch_size=16 \
--test_batch_size=16 \
--cuda=True \
--pretrained_model=model/craft_mlt_25k.pth \
--lr=3e-5 \
--epochs=20 \
--test_interval=40
