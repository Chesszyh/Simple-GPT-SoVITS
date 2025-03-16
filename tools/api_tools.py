# api_v1.py tools
# v2应该有重复的逻辑，不知道他们写的处理方式哪里不同？

import os
import sys
import torch
from typing import List, Tuple
from loguru import logger
import argparse

from ..GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast,load_sovits_new
from text import chinese # FIXME text在哪？没看到

class PretrainedModelTools:
    def __init__(self):
        pass # TODO
    
    def get_phones_and_bert(text,language,version,final=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "all_zh":
                if re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return get_phones_and_bert(formattext,"zh",version)
                else:
                    phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                    bert = get_bert_feature(norm_text, word2ph).to(device)
            elif language == "all_yue" and re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return get_phones_and_bert(formattext,"yue",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if is_half == True else torch.float32,
                ).to(device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist=[]
            langlist=[]
            if language == "auto":
                for tmp in LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
                bert = get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        if not final and len(phones) < 6:
            return get_phones_and_bert("." + text,language,version,final=True)

        return phones,bert.to(torch.float16 if is_half == True else torch.float32),norm_text

    splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
    
    # FIXME 移除这个之前的全局变量
    
    def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, top_k= 15, top_p = 0.6, temperature = 0.6, speed = 1, inp_refs = None, sample_steps = 32, if_sr = False, spk = "default"):
        infer_sovits = speaker_list[spk].sovits
        vq_model = infer_sovits.vq_model
        hps = infer_sovits.hps
        version = vq_model.version

        infer_gpt = speaker_list[spk].gpt
        t2s_model = infer_gpt.t2s_model
        max_sec = infer_gpt.max_sec

        t0 = ttime()
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        prompt_language, text = prompt_language, text.strip("\n")
        dtype = torch.float16 if is_half == True else torch.float32
        zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half == True else np.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if (is_half == True):
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

            if version != "v3":
                refers=[]
                if(inp_refs):
                    for path in inp_refs:
                        try:
                            refer = get_spepc(hps, path).to(dtype).to(device)
                            refers.append(refer)
                        except Exception as e:
                            logger.error(e)
                if(len(refers)==0):
                    refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
            else:
                refer = get_spepc(hps, ref_wav_path).to(device).to(dtype)

        t1 = ttime()
        # os.environ['version'] = version
        prompt_language = dict_language[prompt_language.lower()]
        text_language = dict_language[text_language.lower()]
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
        texts = text.split("\n")
        audio_bytes = BytesIO()

        for text in texts:
            # 简单防止纯符号引发参考音频泄露
            if only_punc(text):
                continue

            audio_opt = []
            if (text[-1] not in splits): text += "。" if text_language != "en" else "."
            phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
            bert = torch.cat([bert1, bert2], 1)

            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            bert = bert.to(device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
            t2 = ttime()
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k = top_k,
                    top_p = top_p,
                    temperature = temperature,
                    early_stop_num=hz * max_sec)
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            t3 = ttime()

            if version != "v3":
                audio = \
                    vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0),
                                    refers,speed=speed).detach().cpu().numpy()[
                        0, 0]  ###试试重建不带上prompt部分
            else:
                phoneme_ids0=torch.LongTensor(phones1).to(device).unsqueeze(0)
                phoneme_ids1=torch.LongTensor(phones2).to(device).unsqueeze(0)
                # print(11111111, phoneme_ids0, phoneme_ids1)
                fea_ref,ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
                ref_audio, sr = torchaudio.load(ref_wav_path)
                ref_audio=ref_audio.to(device).float()
                if (ref_audio.shape[0] == 2):
                    ref_audio = ref_audio.mean(0).unsqueeze(0)
                if sr!=24000:
                    ref_audio=resample(ref_audio,sr)
                # print("ref_audio",ref_audio.abs().mean())
                mel2 = mel_fn(ref_audio)
                mel2 = norm_spec(mel2)
                T_min = min(mel2.shape[2], fea_ref.shape[2])
                mel2 = mel2[:, :, :T_min]
                fea_ref = fea_ref[:, :, :T_min]
                if (T_min > 468):
                    mel2 = mel2[:, :, -468:]
                    fea_ref = fea_ref[:, :, -468:]
                    T_min = 468
                chunk_len = 934 - T_min
                # print("fea_ref",fea_ref,fea_ref.shape)
                # print("mel2",mel2)
                mel2=mel2.to(dtype)
                fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge,speed)
                # print("fea_todo",fea_todo)
                # print("ge",ge.abs().mean())
                cfm_resss = []
                idx = 0
                while (1):
                    fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
                    if (fea_todo_chunk.shape[-1] == 0): break
                    idx += chunk_len
                    fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                    # set_seed(123)
                    cfm_res = vq_model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0)
                    cfm_res = cfm_res[:, :, mel2.shape[2]:]
                    mel2 = cfm_res[:, :, -T_min:]
                    # print("fea", fea)
                    # print("mel2in", mel2)
                    fea_ref = fea_todo_chunk[:, :, -T_min:]
                    cfm_resss.append(cfm_res)
                cmf_res = torch.cat(cfm_resss, 2)
                cmf_res = denorm_spec(cmf_res)
                if bigvgan_model==None:init_bigvgan()
                with torch.inference_mode():
                    wav_gen = bigvgan_model(cmf_res)
                    audio=wav_gen[0][0].cpu().detach().numpy()

            max_audio=np.abs(audio).max()
            if max_audio>1:
                audio/=max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            audio_opt = np.concatenate(audio_opt, 0)
            t4 = ttime()

            sr = hps.data.sampling_rate if version != "v3" else 24000
            if if_sr and sr == 24000:
                audio_opt = torch.from_numpy(audio_opt).float().to(device)
                audio_opt,sr=audio_sr(audio_opt.unsqueeze(0),sr)
                max_audio=np.abs(audio_opt).max()
                if max_audio > 1: audio_opt /= max_audio
                sr = 48000

            if is_int32:
                audio_bytes = pack_audio(audio_bytes,(audio_opt * 2147483647).astype(np.int32),sr)
            else:
                audio_bytes = pack_audio(audio_bytes,(audio_opt * 32768).astype(np.int16),sr)
        # logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
            if stream_mode == "normal":
                audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
                yield audio_chunk
        
        if not stream_mode == "normal": 
            if media_type == "wav":
                sr = 48000 if if_sr else 24000
                sr = hps.data.sampling_rate if version != "v3" else sr
                audio_bytes = pack_wav(audio_bytes,sr)
            yield audio_bytes.getvalue()
    
    def get_bert_feature(text, word2ph):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        # if(is_half==True):phone_level_feature=phone_level_feature.half()
        return phone_level_feature.T


    def clean_text_inf(text, language, version):
        language = language.replace("all_","")
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text


    def get_bert_inf(phones, word2ph, norm_text, language):
        language=language.replace("all_","")
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)

        return bert
    
    
class DefaultRefer:
    def __init__(self, path, text, language):
        self.path = args.default_refer_path
        self.text = args.default_refer_text
        self.language = args.default_refer_language

    def is_ready(self) -> bool:
        return is_full(self.path, self.text, self.language)


def is_empty(*items):  # 任意一项不为空返回False
    for item in items:
        if item is not None and item != "":
            return False
    return True


def is_full(*items):  # 任意一项为空返回False
    for item in items:
        if item is None or item == "":
            return False
    return True

def resample(audio_tensor, sr0):
    global resample_transform_dict
    if sr0 not in resample_transform_dict:
        resample_transform_dict[sr0] = torchaudio.transforms.Resample(
            sr0, 24000
        ).to(device)
    return resample_transform_dict[sr0](audio_tensor)\
        
# NOTE 为什么单独init一个bigvgan？
def init_bigvgan():
    global bigvgan_model
    from BigVGAN import bigvgan
    bigvgan_model = bigvgan.BigVGAN.from_pretrained("%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,), use_cuda_kernel=False)  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)
    
# 对话人
class Speaker:
    def __init__(self, name, gpt, sovits, phones = None, bert = None, prompt = None):
        self.name = name
        self.sovits = sovits
        self.gpt = gpt
        self.phones = phones
        self.bert = bert
        self.prompt = prompt

# SoVITS
# 就这两个b成员变量还要封个类？Why? NOTE
class Sovits:
    def __init__(self, vq_model, hps):
        self.vq_model = vq_model
        self.hps = hps
        
def get_sovits_weights(sovits_path):
    path_sovits_v3="GPT_SoVITS/pretrained_models/s2Gv3.pth"
    is_exist_s2gv3=os.path.exists(path_sovits_v3)

    version, model_version, if_lora_v3=get_sovits_version_from_path_fast(sovits_path)
    if if_lora_v3==True and is_exist_s2gv3==False:
        logger.info("SoVITS V3 底模缺失，无法加载相应 LoRA 权重")

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if 'enc_p.text_embedding.weight' not in dict_s2['weight']:
        hps.model.version = "v2"#v3model,v2sybomls
    elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"

    if model_version == "v3":
        hps.model.version = "v3"

    model_params_dict = vars(hps.model)
    if model_version!="v3":
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model_params_dict
        )
    else:
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model_params_dict
        )
        init_bigvgan()
    model_version=hps.model.version
    logger.info(f"模型版本: {model_version}")
    if ("pretrained" not in sovits_path):
        try:
            del vq_model.enc_q
        except:pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    if if_lora_v3 == False:
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
    else:
        vq_model.load_state_dict(load_sovits_new(path_sovits_v3)["weight"], strict=False)
        lora_rank=dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        # torch.save(vq_model.state_dict(),"merge_win.pth")
        vq_model.eval()

    sovits = Sovits(vq_model, hps)
    return sovits

class Gpt:
    def __init__(self, max_sec, t2s_model):
        self.max_sec = max_sec
        self.t2s_model = t2s_model
        
def get_gpt_weights(gpt_path):
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    # total = sum([param.nelement() for param in t2s_model.parameters()])
    # logger.info("Number of parameter: %.2fM" % (total / 1e6))

    gpt = Gpt(max_sec, t2s_model)
    return gpt

def change_gpt_sovits_weights(gpt_path,sovits_path):
    try:
        gpt = get_gpt_weights(gpt_path)
        sovits = get_sovits_weights(sovits_path)
    except Exception as e:
        return JSONResponse({"code": 400, "message": str(e)}, status_code=400)

    speaker_list["default"] = Speaker(name="default", gpt=gpt, sovits=sovits)
    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)

# 事件处理
class EventHandler:
    def __init__(self, name, event, handler):
        self.name = name
        self.event = event
        self.handler = handler
    
    # 从api.py的0级缩进处剪切过来的
    def handle_control(command):
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


    def handle_change(path, text, language):
        if is_empty(path, text, language):
            return JSONResponse({"code": 400, "message": '缺少任意一项以下参数: "path", "text", "language"'}, status_code=400)

        if path != "" or path is not None:
            default_refer.path = path
        if text != "" or text is not None:
            default_refer.text = text
        if language != "" or language is not None:
            default_refer.language = language

        logger.info(f"当前默认参考音频路径: {default_refer.path}")
        logger.info(f"当前默认参考音频文本: {default_refer.text}")
        logger.info(f"当前默认参考音频语种: {default_refer.language}")
        logger.info(f"is_ready: {default_refer.is_ready()}")


        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


    def handle(refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr):
        if (
                refer_wav_path == "" or refer_wav_path is None
                or prompt_text == "" or prompt_text is None
                or prompt_language == "" or prompt_language is None
        ):
            refer_wav_path, prompt_text, prompt_language = (
                default_refer.path,
                default_refer.text,
                default_refer.language,
            )
            if not default_refer.is_ready():
                return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)

        if not sample_steps in [4,8,16,32]:
            sample_steps = 32

        if cut_punc == None:
            text = cut_text(text,default_cut_punc)
        else:
            text = cut_text(text,cut_punc)

        return StreamingResponse(get_tts_wav(refer_wav_path, prompt_text, prompt_language, text, text_language, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr), media_type="audio/"+media_type)


class AudioProcesser:
    def __init__():
        pass
    
    def get_spepc(hps, filename):
        audio,_ = librosa.load(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx=audio.abs().max()
        if(maxx>1):
            audio/=min(2,maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                                hps.data.win_length, center=False)
        return spec


    def pack_audio(audio_bytes, data, rate):
        if media_type == "ogg":
            audio_bytes = pack_ogg(audio_bytes, data, rate)
        elif media_type == "aac":
            audio_bytes = pack_aac(audio_bytes, data, rate)
        else:
            # wav无法流式, 先暂存raw
            audio_bytes = pack_raw(audio_bytes, data, rate)

        return audio_bytes


    def pack_ogg(audio_bytes, data, rate):
        # Author: AkagawaTsurunaki
        # Issue:
        #   Stack overflow probabilistically occurs
        #   when the function `sf_writef_short` of `libsndfile_64bit.dll` is called
        #   using the Python library `soundfile`
        # Note:
        #   This is an issue related to `libsndfile`, not this project itself.
        #   It happens when you generate a large audio tensor (about 499804 frames in my PC)
        #   and try to convert it to an ogg file.
        # Related:
        #   https://github.com/RVC-Boss/GPT-SoVITS/issues/1199
        #   https://github.com/libsndfile/libsndfile/issues/1023
        #   https://github.com/bastibe/python-soundfile/issues/396
        # Suggestion:
        #   Or split the whole audio data into smaller audio segment to avoid stack overflow?

        def handle_pack_ogg():
            with sf.SoundFile(audio_bytes, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
                audio_file.write(data)

        import threading
        # See: https://docs.python.org/3/library/threading.html
        # The stack size of this thread is at least 32768
        # If stack overflow error still occurs, just modify the `stack_size`.
        # stack_size = n * 4096, where n should be a positive integer.
        # Here we chose n = 4096.
        stack_size = 4096 * 4096
        try:
            threading.stack_size(stack_size)
            pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
            pack_ogg_thread.start()
            pack_ogg_thread.join()
        except RuntimeError as e:
            # If changing the thread stack size is unsupported, a RuntimeError is raised.
            print("RuntimeError: {}".format(e))
            print("Changing the thread stack size is unsupported.")
        except ValueError as e:
            # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
            print("ValueError: {}".format(e))
            print("The specified stack size is invalid.")

        return audio_bytes


    def pack_raw(audio_bytes, data, rate):
        audio_bytes.write(data.tobytes())

        return audio_bytes


    def pack_wav(audio_bytes, rate):
        if is_int32:
            data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int32)
            wav_bytes = BytesIO()
            sf.write(wav_bytes, data, rate, format='WAV', subtype='PCM_32')
        else:
            data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int16)
            wav_bytes = BytesIO()
            sf.write(wav_bytes, data, rate, format='WAV')
        return wav_bytes


    def pack_aac(audio_bytes, data, rate):
        if is_int32:
            pcm = 's32le'
            bit_rate = '256k'
        else:
            pcm = 's16le'
            bit_rate = '128k'
        process = subprocess.Popen([
            'ffmpeg',
            '-f', pcm,  # 输入16位有符号小端整数PCM
            '-ar', str(rate),  # 设置采样率
            '-ac', '1',  # 单声道
            '-i', 'pipe:0',  # 从管道读取输入
            '-c:a', 'aac',  # 音频编码器为AAC
            '-b:a', bit_rate,  # 比特率
            '-vn',  # 不包含视频
            '-f', 'adts',  # 输出AAC数据流格式
            'pipe:1'  # 将输出写入管道
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate(input=data.tobytes())
        audio_bytes.write(out)

        return audio_bytes


    def read_clean_buffer(audio_bytes):
        audio_chunk = audio_bytes.getvalue()
        audio_bytes.truncate(0)
        audio_bytes.seek(0)

        return audio_bytes, audio_chunk


    def cut_text(text, punc):
        punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
        if len(punc_list) > 0:
            punds = r"[" + "".join(punc_list) + r"]"
            text = text.strip("\n")
            items = re.split(f"({punds})", text)
            mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
            # 在句子不存在符号或句尾无符号的时候保证文本完整
            if len(items)%2 == 1:
                mergeitems.append(items[-1])
            text = "\n".join(mergeitems)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        return text

    def only_punc(text):
        return not any(t.isalnum() or t.isalpha() for t in text)
    
    
class DictToAttrRecursive(dict):
    """处理配置文件或需要以面向对象的方式访问字典数据的情况"""
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")