# backend/src/tts_stt.py
import os
import tempfile
import io
import base64
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

def translate_text(text, target_lang):
    """Translates text to the target language."""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

def tts(text, lang="en"):
    """
    Generates text-to-speech audio and returns it as a Base64 data URL.
    """
    try:
        speech = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        speech.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        mp3_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
        audio_url = f"data:audio/mpeg;base64,{mp3_base64}"
        return {"status": "ok", "audio_url": audio_url}
    except Exception as e:
        print(f"TTS Error: {e}")
        return {"status": "error", "message": str(e)}

def stt(audio_file, lang="en"):
    """
    Transcribes audio from a file object.
    """
    lang_map = {
        "en": "en-US", "hi": "hi-IN", "kn": "kn-IN", "ta": "ta-IN", "es": "es-ES"
    }
    api_lang = lang_map.get(lang, "en-US")
    
    r = sr.Recognizer()
    try:
        # Use pydub to convert from any format
        audio_segment = AudioSegment.from_file(audio_file.stream)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f_wav:
            audio_segment.export(f_wav.name, format="wav")
            with sr.AudioFile(f_wav.name) as source:
                audio_data = r.record(source)
            text = r.recognize_google(audio_data, language=api_lang)
            return {"text": text}
            
    except sr.UnknownValueError:
        return {"text": "", "error": "Could not understand audio"}
    except sr.RequestError as e:
        return {"text": "", "error": f"API error: {e}"}
    except Exception as e:
        print(f"STT Error: {e}")
        return {"text": "", "error": f"Failed to process audio: {e}"}