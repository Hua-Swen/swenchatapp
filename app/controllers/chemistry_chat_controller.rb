# frozen_string_literal: true

class ChemistryChatController < ApplicationController
  protect_from_forgery with: :null_session, only: [:create]

  def show
  end

  # POST /chemistry_chat/message
  def create
    user_message = params[:message].to_s.strip
    k = (params[:k].presence || 5).to_i

    if user_message.blank?
      render json: { error: "Message cannot be empty." }, status: :unprocessable_entity
      return
    end

    rag_result = RagClient.new.ask(query: user_message, k: k)

    render json: {
      reply: rag_result["answer"],
      sources: rag_result["sources"] || []
    }
  rescue RagClient::RagError => e
    render json: { error: e.message }, status: :bad_gateway
  rescue StandardError => e
    render json: { error: "Unexpected error: #{e.message}" }, status: :internal_server_error
  end
end
