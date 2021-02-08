cd D:\Documents\GitHub\EfficientNet\models\official\efficientnet
python main.py ^
--mode="train_and_eval" ^
--train_steps=170614  ^
--train_batch_size=50  ^
--eval_batch_size=50 ^
--num_train_images=170614  ^
--num_eval_images=27680  ^
--steps_per_eval=25  ^
--iterations_per_loop=25  ^
--num_label_classes=10 ^
--use_tpu=False ^
--data_dir="D:\TPU\data" ^
--model_dir="D:\TPU\model" ^
--export_dir="D:\TPU\export" ^
--model_name="efficientnet-bx" ^
--input_image_size=15 ^
--tpu="" ^
--use_bfloat16="False" ^
--data_format="channels_last"
pause
