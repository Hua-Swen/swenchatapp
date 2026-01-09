# frozen_string_literal: true

require "net/http"
require "json"
require "uri"

class RagClient
  class RagError < StandardError; end

  def initialize(base_url: ENV.fetch("RAG_SERVICE_URL", "http://localhost:8811"))
    @base_url = base_url
  end

  def ask(query:, k: 5)
    uri = URI.join(@base_url, "/rag/ask")

    request = Net::HTTP::Post.new(uri)
    request["Content-Type"] = "application/json"
    request.body = { query: query, k: k }.to_json

    response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: uri.scheme == "https") do |http|
      http.read_timeout = 180
      http.open_timeout = 10
      http.request(request)
    end

    unless response.is_a?(Net::HTTPSuccess)
      raise RagError, "RAG service error: #{response.code} #{response.body}"
    end

    JSON.parse(response.body)
  end
end
