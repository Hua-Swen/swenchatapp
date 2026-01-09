require "open3"
require "tempfile"

class SpeechController < ApplicationController
  protect_from_forgery with: :exception

  def create
    audio = params[:audio]
    return render json: { error: "No audio uploaded" }, status: :bad_request if audio.blank?

    input = Tempfile.new(["upload", File.extname(audio.original_filename.presence || ".webm")])
    input.binmode
    input.write(audio.read)
    input.flush

    wav = Tempfile.new(["audio", ".wav"])
    wav.close

    ffmpeg_cmd = [
      "ffmpeg", "-y",
      "-i", input.path,
      "-ar", "16000",
      "-ac", "1",
      "-c:a", "pcm_s16le",
      wav.path
    ]

    _out, err, status = Open3.capture3(*ffmpeg_cmd)
    unless status.success?
      Rails.logger.error("ffmpeg failed: #{err}")
      return render json: { error: "Audio conversion failed" }, status: :unprocessable_entity
    end

    whisper_cli = ENV.fetch("WHISPER_CLI_PATH")
    model_path  = ENV.fetch("WHISPER_MODEL_PATH")

    whisper_cmd = [
      whisper_cli,
      "-m", model_path,
      "-f", wav.path,
      "-nt",
      "-l", "en"
    ]

    out, werr, wstatus = Open3.capture3(*whisper_cmd)
    unless wstatus.success?
      Rails.logger.error("whisper failed: #{werr}\n#{out}")
      return render json: { error: "Transcription failed" }, status: :unprocessable_entity
    end

    transcript = extract_transcript(out)
    render json: { transcript: transcript }
  ensure
    input&.close! rescue nil
    wav&.close! rescue nil
  end

  private

  def extract_transcript(stdout)
    lines = stdout.to_s.split("\n").map(&:strip).reject(&:empty?)
    lines.last(5).join(" ").strip
  end
end
