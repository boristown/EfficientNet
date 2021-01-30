python main.py ^
--train_steps=100000  ^
--train_batch_size=10  ^
--eval_batch_size=10 ^
--num_train_images=45584  ^
--num_eval_images=4312  ^
--steps_per_eval=10  ^
--iterations_per_loop=10  ^
--use_tpu=False ^
--data_dir="D:\TPU\data" ^
--model_dir="D:\TPU\model" ^
--export_dir="D:\TPU\export" ^
--model_name="efficientnet-bx" ^
--tpu="" ^
--use_bfloat16="False" ^
--data_format="channels_last"
pause
