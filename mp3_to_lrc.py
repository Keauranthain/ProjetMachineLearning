import time
import warnings
import os
import configparser
import torch
from faster_whisper import WhisperModel
from jiwer import wer
import re

def calculate_precision(wts_file, verification_file):
    with open(wts_file, "r", encoding="utf-8") as wts:
        transcribed_text = wts.read()

    with open(verification_file, "r", encoding="utf-8") as vf:
        reference_text = vf.read()

    # Calculate WER, ignore punctuation and structure
    error_rate = wer(reference_text, transcribed_text)
    return error_rate

def transcribe_to_lrc(
        songs: list,
        model_size: str,
        beam_size: int,
        best_of: int,
        language: str,
        temperature: float,
        verbose: bool,
        initial_prompt: str,
        output_format: str
):
    start_time = time.time()
    total_accurency = 0.0
    songs_for_accurency = 0
    Theme_accurency = []

    def stamp():
        return f"[{(time.time() - start_time):.2f}s]"

    print(f"{stamp()} Paramètres de lancement : "
          f"audio_to_translate={songs}, "
          f"model_size={model_size}, "
          f"beam_size={beam_size}, "
          f"best_of={best_of}, "
          f"language={language}, "
          f"temperature={temperature}, "
          f"verbose={verbose}")

    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False")
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")



    print(f"{stamp()} Chargement du modèle Faster Whisper ({model_size})...")
    if torch.cuda.is_available():
        print(f"{stamp()} GPU running...")
        device = "cuda"
    else:
        print(f"{stamp()} CPU running...")
        device = "cpu"

    model = WhisperModel(model_size, device=device, compute_type="float32")

    for song in songs:
        result_folder = config.get("Paths", "result_folder")
        verification_folder = config.get("Paths", "verification_folder", fallback="Verification")
        wts_folder = config.get("Paths", "wts_folder", fallback="Results_WTS")
        audio_file = os.path.join(music_folder, f"{song}.mp3")
        lrc_file = os.path.join(result_folder, f"{song}.lrc")
        wts_file = os.path.join(wts_folder, f"{song}.txt")
        verification_file = os.path.join(verification_folder, f"{song}.lrc")

        if not os.path.isfile(audio_file):
            raise FileNotFoundError(f"{stamp()} Fichier audio introuvable : {audio_file}")

        result_folder = os.path.dirname(lrc_file)
        os.makedirs(result_folder, exist_ok=True)
        wts_folder = os.path.dirname(wts_file)
        os.makedirs(wts_folder, exist_ok=True)

        print(f"{stamp()} Transcription du fichier {audio_file} en cours...")
        segments, _ = model.transcribe(
            audio_file,
            beam_size=beam_size,
            best_of=best_of,
            language=language,
            temperature=temperature,
            initial_prompt=initial_prompt
        )

        segments = list(segments)  # Convert generator to list

        print(f"{stamp()} Transcription terminée. Nombre de segments : {len(segments)}")

        with open(lrc_file, "w", encoding="utf-8") as f:
            for seg in segments:
                start = seg.start
                text = seg.text.strip()
                minutes = int(start // 60)
                seconds = int(start % 60)
                hundredths = int((start * 100) % 100)

                timestamp = f"[{minutes:02d}:{seconds:02d}.{hundredths:02d}]"
                f.write(f"{timestamp} {text}\n")

        print(f"{stamp()} Fichier {output_format.upper()} généré : {lrc_file}")

        with open(wts_file, "w", encoding="utf-8") as f:
            for seg in segments:
                text = seg.text.strip()
                f.write(f"{text}\n")

        print(f"{stamp()} Fichier WTS généré : {wts_file}")

        if os.path.isfile(verification_file):
            precision = calculate_precision(wts_file, verification_file)
            total_accurency += precision
            songs_for_accurency += 1

            match = re.search(r'^(.*?)__', song).group(1)
            not_exist = True
            for theme in Theme_accurency:
                if theme[0] == match:
                    not_exist = False
                    theme[1]+=precision
                    theme[2]+=precision
            if not_exist:
                Theme_accurency.append([match, precision,1])

            print(f"{stamp()} WER : {precision}")
        else:
            print(f"{stamp()} Fichier de vérification non trouvé : {verification_file}")
    print(f"\n\n{stamp()} Total précision : {total_accurency/songs_for_accurency}")
    if len(Theme_accurency)>1:
        for theme in Theme_accurency:
            print(f"{stamp()} {theme[0]} : {theme[1]/theme[2]}")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("settings.ini")

    song_name = config.get("Files", "song_name")
    music_folder = config.get("Paths", "music_folder")

    song_list = []
    if song_name.lower() == "all":
        for file in os.listdir(music_folder):
            song_list.append(os.path.splitext(file)[0])
    elif song_name.lower().startswith("all__"):
        filters = song_name.split("__")[1:]
        pattern = re.compile(r'^(' + '|'.join(map(re.escape, filters)) + r')__', re.IGNORECASE)
        for file in os.listdir(music_folder):
            basename = os.path.splitext(file)[0]
            if pattern.search(basename):
                song_list.append(basename)
    else:
        song_list = [song_name]

    transcribe_to_lrc(
        songs=song_list,
        model_size=config.get("Whisper", "model_size"),
        beam_size=config.getint("Whisper", "beam_size"),
        best_of=config.getint("Whisper", "best_of"),
        language=None if config.get("Whisper", "language").lower() == "none" else config.get("Whisper", "language"),
        temperature=config.getfloat("Whisper", "temperature"),
        verbose=config.getboolean("Whisper", "verbose"),
        initial_prompt=config.get("Whisper", "initial_prompt"),
        output_format=config.get("Whisper", "output_format")
    )
