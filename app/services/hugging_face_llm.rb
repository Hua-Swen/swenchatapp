# app/services/hugging_face_llm.rb
require 'net/http'
require 'uri'
require 'json'
require 'openssl'

class HuggingFaceLlm
  # OpenAI-compatible Chat Completions endpoint via HF Router
  API_URL  = "https://router.huggingface.co/v1/chat/completions".freeze

  # You can optionally add a provider suffix like ":scaleway" later if needed.
  MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct".freeze

  class << self
    # messages: [{ role: "user"|"assistant"|"system", content: "..." }, ...]
    def chat(system_prompt:, messages:)
      uri  = URI(API_URL)
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = true

      # Use your CA bundle if configured (we already set SSL_CERT_FILE)
      if ENV["SSL_CERT_FILE"].present?
        store = OpenSSL::X509::Store.new
        store.set_default_paths
        store.add_file(ENV["SSL_CERT_FILE"])
        http.cert_store = store
      end

      request = Net::HTTP::Post.new(uri)
      request["Authorization"]  = "Bearer #{ENV.fetch('HUGGING_FACE_API_TOKEN')}"
      request["Content-Type"]   = "application/json"

      # Build OpenAI-style chat completion payload
      full_messages = []
      full_messages << { role: "system", content: system_prompt } if system_prompt.present?
      full_messages.concat(messages)

      body = {
        model:    MODEL_ID,
        messages: full_messages,
        max_tokens: 256,
        temperature: 0.3,
        top_p: 0.9
      }

      request.body = body.to_json

      response = http.request(request)

      unless response.is_a?(Net::HTTPSuccess)
        Rails.logger.error("HuggingFace error: #{response.code} - #{response.body}")
        raise "HuggingFace API error: #{response.code}"
      end

      parse_response(JSON.parse(response.body))
    end

    private

    # HF Router uses OpenAI-compatible chat completion format:
    # {
    #   "choices": [
    #     {
    #       "message": { "role": "assistant", "content": "..." },
    #       ...
    #     }
    #   ],
    #   ...
    # }
    def parse_response(json)
      choice = json.dig("choices", 0, "message", "content")
      return choice.strip if choice.is_a?(String)

      Rails.logger.warn("Unexpected HF router response: #{json.inspect}")
      json.to_s
    end
  end
end
