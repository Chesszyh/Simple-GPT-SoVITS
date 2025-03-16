import requests

url = "http://localhost:9880/tts"
params = {
    "prompt_lang": "en",
    "prompt_text": "Hello. What's wrong with you? Fuck you!",
    "text_split_method": "cut5",
    "batch_size": 1,
    "media_type": "wav",
    "streaming_mode": "true"
}

try:
    response = requests.get(url, params=params, stream=True)
    response.raise_for_status()  # 检查请求是否成功

    if response.headers['Content-Type'] == 'audio/wav':
        with open("output.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("音频已保存到 output.wav")
    else:
        print("错误：返回的不是音频数据")
        print(response.json())

except requests.exceptions.RequestException as e:
    print(f"请求出错: {e}")