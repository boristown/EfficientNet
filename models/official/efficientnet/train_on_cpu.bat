python main.py ^
--train_steps=100000  ^
--train_batch_size=64  ^
--eval_batch_size=64  ^
--num_train_images=3313529  ^
--num_eval_images=17318  ^
--steps_per_eval=500  ^
--iterations_per_loop=500  ^
--use_tpu=False ^
--data_dir="D:\TPU\data" ^
--model_dir="D:\TPU\model" ^
--export_dir="D:\TPU\export" ^
--model_name="efficientnet-x0" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last"
pause
