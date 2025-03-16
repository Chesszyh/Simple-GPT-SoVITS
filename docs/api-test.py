from gradio_client import Client

client = Client("http://localhost:9874/")
result = client.predict(
		batch_size=4,
		total_epoch=8,
		exp_name="xxx",
		text_low_lr_rate=0.4,
		if_save_latest=True,
		if_save_every_weights=True,
		save_every_epoch=4,
		gpu_numbers1Ba="0",
		pretrained_s2G="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
		pretrained_s2D="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
		if_grad_ckpt=False,
		lora_rank="32",
		api_name="/open1Ba"
)
print(result)