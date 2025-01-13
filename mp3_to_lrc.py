import time
import warnings
import os
import configparser
import torch
from faster_whisper import WhisperModel

def transcribe_to_lrc(
        audio_file: str,
        lrc_file: str,
        wts_file: str,
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

    def stamp():
        return f"[{(time.time() - start_time):.2f}s]"

    print(f"{stamp()} Paramètres de lancement : "
          f"audio_file={audio_file}, "
          f"lrc_file={lrc_file}, "
          f"wts_file={wts_file}, "
          f"model_size={model_size}, "
          f"beam_size={beam_size}, "
          f"best_of={best_of}, "
          f"language={language}, "
          f"temperature={temperature}, "
          f"verbose={verbose}")

    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False")
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"{stamp()} Fichier audio introuvable : {audio_file}")

    # Create result folders if they don't exist
    result_folder = os.path.dirname(lrc_file)
    os.makedirs(result_folder, exist_ok=True)
    wts_folder = os.path.dirname(wts_file)
    os.makedirs(wts_folder, exist_ok=True)

    print(f"{stamp()} Chargement du modèle Faster Whisper ({model_size})...")
    try:
        model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
    except ValueError as e:
        print(f"{stamp()} Erreur lors du chargement du modèle : {e}")
        print(f"{stamp()} Rechargement avec 'float32'...")
        model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float32")

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

    # Write LRC file with timestamps
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

    # Write WTS file without timestamps
    with open(wts_file, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg.text.strip()
            f.write(f"{text}\n")

    print(f"{stamp()} Fichier WTS généré : {wts_file}")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("settings.ini")

    song_name = config.get("Files", "song_name")
    music_folder = config.get("Paths", "music_folder")
    result_folder = config.get("Paths", "result_folder")
    wts_folder = config.get("Paths", "wts_folder", fallback="Results_WTS")

    if song_name.lower() == "all":
        for file in os.listdir(music_folder):
            if file.endswith(".mp3"):
                song_name = os.path.splitext(file)[0]
                audio_file = os.path.join(music_folder, file)
                lrc_file = os.path.join(result_folder, f"{song_name}.lrc")
                wts_file = os.path.join(wts_folder, f"{song_name}.txt")

                transcribe_to_lrc(
                    audio_file=audio_file,
                    lrc_file=lrc_file,
                    wts_file=wts_file,
                    model_size=config.get("Whisper", "model_size"),
                    beam_size=config.getint("Whisper", "beam_size"),
                    best_of=config.getint("Whisper", "best_of"),
                    language=None if config.get("Whisper", "language").lower() == "none" else config.get("Whisper", "language"),
                    temperature=config.getfloat("Whisper", "temperature"),
                    verbose=config.getboolean("Whisper", "verbose"),
                    initial_prompt=config.get("Whisper", "initial_prompt"),
                    output_format=config.get("Whisper", "output_format")
                )
    else:
        audio_file = os.path.join(music_folder, f"{song_name}.mp3")
        lrc_file = os.path.join(result_folder, f"{song_name}.lrc")
        wts_file = os.path.join(wts_folder, f"{song_name}.txt")

        transcribe_to_lrc(
            audio_file=audio_file,
            lrc_file=lrc_file,
            wts_file=wts_file,
            model_size=config.get("Whisper", "model_size"),
            beam_size=config.getint("Whisper", "beam_size"),
            best_of=config.getint("Whisper", "best_of"),
            language=None if config.get("Whisper", "language").lower() == "none" else config.get("Whisper", "language"),
            temperature=config.getfloat("Whisper", "temperature"),
            verbose=config.getboolean("Whisper", "verbose"),
            initial_prompt=config.get("Whisper", "initial_prompt"),
            output_format=config.get("Whisper", "output_format")
        )
