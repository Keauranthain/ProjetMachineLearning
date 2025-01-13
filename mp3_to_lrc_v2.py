import time
import warnings
import whisper
import os
from whisper.utils import WriteSRT
from whisper.normalizers import BasicTextNormalizer

def transcribe_to_lrc(
        audio_file: str = "tls.mp3",
        lrc_file: str = "output.lrc",
        model_size: str = "medium",
        beam_size=1,
        best_of=5,
        language=None,
        temperature=0.2,
        verbose=False,
        start_time=None,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.0,
        initial_prompt=None,
        output_format="lrc"
):
    if start_time is None:
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
    transcribe_to_lrc()
