from gradio_client import Client, file

client = Client("http://localhost:9872/")
result = client.predict(
		ref_wav_path="D:\\Git\\Project\\Neuro-sama\\audio\\Simple-GPT-SoVITS\\test\\neuro_ref_1.wav",
		prompt_text="They were so cute. I wanted to pet them but I was too scared.",
		prompt_language="English",
		text="Hello. What's wrong with you? Fuck you!",
		text_language="English",
		how_to_cut="Slice once every 4 sentences",
		top_k=15,
		top_p=1,
		temperature=1,
		ref_free=False,
		speed=1,
		if_freeze=False,
		inp_refs=None,
		sample_steps="32",
		if_sr=False,
		pause_second=0.3,
		api_name="/get_tts_wav"
)
print(result)