from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Generator
import anthropic
from datetime import datetime
import json
import sys
import os

@dataclass
class ClaudeConfig:
    """Configuration settings for Claude chat."""
    max_tokens: int = 1024
    temperature: float = 0.8
    model: str = "claude-3-5-sonnet-latest"
    system_prompt: str = "You are a good Claude!"

def show_help() -> None:
    """Display available commands."""
    print("Available commands:")
    print("/sys <new_prompt> - Change the system prompt")
    print("/temp <value> - Change the temperature")
    print("/max <value> - Change the max tokens")
    print("/env - Show the current environment settings")
    print("/new - Clear the conversation history")
    print("/exit, /quit, /bye - Exit the chat")
    print("/help - Show this help")

def read_multiline_input(prompt: str) -> str:
    """Read potentially multiline input from user."""
    print(prompt, end="")
    try:
        line = input()
    except EOFError:
        sys.exit()

    if not line.startswith("\\"):
        return line

    lines = [line]
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        print("\n<end of input>\n")
        return "\n".join(lines)

def ask_claude(
    messages: List[Dict[str, str]], 
    config: ClaudeConfig
) -> Generator[anthropic.types.MessageStreamEvent, None, None]:
    """Send a message to Claude and get streaming response."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages_formatted = [
        {"role": "assistant" if msg["role"] == "assistant" else "user", "content": msg["content"]}
        for msg in messages[1:]  # Skip system message
    ]

    return client.messages.create(
        model=config.model,
        system=messages[0]["content"],
        messages=messages_formatted,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stream=True
    )

def process_commands(
    user_input: str, 
    conversation_history: List[Dict[str, str]], 
    config: ClaudeConfig
) -> Tuple[Optional[bool], List[Dict[str, str]], ClaudeConfig]:
    """Process user commands and update config/history accordingly."""
    if user_input.startswith("/sys"):
        conversation_history[0]["content"] = user_input[5:].strip()
        print(f"System prompt changed to: {conversation_history[0]['content']}")
    elif user_input.startswith("/temp"):
        config.temperature = float(user_input[5:].strip())
        print(f"Temperature changed to: {config.temperature}")
    elif user_input.startswith("/max"):
        config.max_tokens = int(user_input[4:].strip())
        print(f"Max tokens changed to: {config.max_tokens}")
    elif user_input.lower() == "/env":
        print(f"System prompt: {conversation_history[0]['content']}")
        print(f"Temperature: {config.temperature}")
        print(f"Max tokens: {config.max_tokens}")
    elif user_input.lower() == "/save":
        with open('claude_hist.txt', 'a') as file:
            file.write("\n" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
            json.dump(conversation_history, file, indent=2)
            file.write("\n")
        conversation_history = [conversation_history[0]]
        print("Conversation appended to log")
    elif user_input.lower() == "/new":
        conversation_history = [conversation_history[0]]
        print("Conversation history cleared.")
    elif user_input.lower().startswith("/help") or user_input == '':
        show_help()
    elif user_input.lower() in ["/exit", "/quit", "/bye"]:
        print("Exiting gracefully...")
        sys.exit(0)
    else:
        return None, conversation_history, config

    return True, conversation_history, config

def chat() -> None:
    """Main chat loop."""
    config = ClaudeConfig()
    conversation_history = [{"role": "system", "content": config.system_prompt}]

    while True:
        try:
            print()
            user_input = read_multiline_input("You: ")
            
            # Handle commands
            command_result, conversation_history, config = process_commands(
                user_input, conversation_history, config
            )
            if command_result is not None:
                continue

            # Normal message handling
            conversation_history.append({"role": "user", "content": user_input})
            response_stream = ask_claude(conversation_history, config)

            # Process response
            assistant_reply = ""
            print()
            for chunk in response_stream:
                if chunk.type == "content_block_delta":
                    chunk_content = chunk.delta.text
                    assistant_reply += chunk_content
                    print(chunk_content, end="", flush=True)

            conversation_history.append({
                "role": "assistant",
                "content": assistant_reply + "\n"
            })
            print("\n")

        except EOFError:
            print("Closing...")
            sys.exit()
        except KeyboardInterrupt:
            print("Interrupted!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat()