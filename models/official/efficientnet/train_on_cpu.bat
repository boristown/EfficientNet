python main.py ^
--mode="train_and_eval" ^
--train_steps=45584  ^
--train_batch_size=32  ^
--eval_batch_size=32 ^
--num_train_images=45584  ^
--num_eval_images=4312  ^
--steps_per_eval=50  ^
--iterations_per_loop=50  ^
--num_label_classes=5 ^
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
