class ChatController < ApplicationController
  protect_from_forgery except: :create

  def show
    # Just rendering the chat view
  end

  def create
    user_message = params[:message].to_s
    history      = params[:history] || []

    system_prompt = "You are a helpful AI assistant built into a Ruby on Rails application. " \
                    "Answer clearly and concisely. If the user asks about Rails, feel free to give code."

    messages = history.map do |m|
      {
        role:    m[:role] || m["role"],
        content: m[:content] || m["content"]
      }
    end

    messages << { role: "user", content: user_message }

    begin
      ai_reply = HuggingFaceLlm.chat(
        system_prompt: system_prompt,
        messages:      messages
      )

      messages << { role: "assistant", content: ai_reply }

      render json: {
        reply:   ai_reply,
        history: messages
      }
    rescue => e
      Rails.logger.error("Chat error: #{e.class} - #{e.message}")
      render json: { error: "Something went wrong talking to the AI." }, status: :internal_server_error
    end
  end
end
