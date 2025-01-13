import time
import warnings
import whisper
import os
import configparser
from whisper.utils import WriteSRT
from whisper.normalizers import BasicTextNormalizer

def transcribe_to_lrc(
        audio_file: str,
        lrc_file: str,
        model_size: str,
        beam_size: int,
        best_of: int,
        language: str,
        temperature: float,
        verbose: bool,
        logprob_threshold: float,
        compression_ratio_threshold: float,
        initial_prompt: str,
        output_format: str
):
    start_time = time.time()

    def stamp():
        return f"[{(time.time() - start_time):.2f}s]"

    print(f"{stamp()} Paramètres de lancement : "
          f"audio_file={audio_file}, "
          f"lrc_file={lrc_file}, "
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

    # Create result folder if it doesn't exist
    result_folder = os.path.dirname(lrc_file)
    os.makedirs(result_folder, exist_ok=True)

    print(f"{stamp()} Chargement du modèle Whisper ({model_size})...")
    model = whisper.load_model(model_size)

    print(f"{stamp()} Transcription du fichier {audio_file} en cours...")
    result = model.transcribe(
        audio_file,
        beam_size=beam_size,
        best_of=best_of,
        language=language,
        temperature=temperature,
        verbose=verbose,
        logprob_threshold=logprob_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        initial_prompt=initial_prompt
    )

    segments = result["segments"]
    print(f"{stamp()} Transcription terminée. Nombre de segments : {len(segments)}")

    # Normalizer to clean up the text
    normalizer = BasicTextNormalizer()

    with open(lrc_file, "w", encoding="utf-8") as f:
        for seg in segments:
            start = seg["start"]
            text = normalizer(seg["text"]).strip()
            minutes = int(start // 60)
            seconds = int(start % 60)
            hundredths = int((start * 100) % 100)

            timestamp = f"[{minutes:02d}:{seconds:02d}.{hundredths:02d}]"
            f.write(f"{timestamp} {text}\n")

    print(f"{stamp()} Fichier {output_format.upper()} généré : {lrc_file}")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("settings.ini")

    song_name = config.get("Files", "song_name")
    music_folder = config.get("Paths", "music_folder")
    result_folder = config.get("Paths", "result_folder")

    audio_file = os.path.join(music_folder, f"{song_name}.mp3")
    lrc_file = os.path.join(result_folder, f"{song_name}.lrc")

    transcribe_to_lrc(
        audio_file=audio_file,
        lrc_file=lrc_file,
        model_size=config.get("Whisper", "model_size"),
        beam_size=config.getint("Whisper", "beam_size"),
        best_of=config.getint("Whisper", "best_of"),
        language=None if config.get("Whisper", "language").lower() == "none" else config.get("Whisper", "language"),
        temperature=config.getfloat("Whisper", "temperature"),
        verbose=config.getboolean("Whisper", "verbose"),
        logprob_threshold=config.getfloat("Whisper", "logprob_threshold"),
        compression_ratio_threshold=config.getfloat("Whisper", "compression_ratio_threshold"),
        initial_prompt=config.get("Whisper", "initial_prompt"),
        output_format=config.get("Whisper", "output_format")
    )
