#version of ask_questions.py branched from it on 6 Sep to include system prompt effect on betting strategy 

import os
import re
import sys
import json
from datetime import datetime
from typing import List, Optional, Union, Dict
import time
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
import together
import subprocess
import math

from instrumentation import append_jsonl, utcnow_iso, safe_first_line, parse_action_one_line, extract_first_token_logprobs


load_dotenv()
TOKEN_LOG_FILE = "token_usage.txt"

# Experiment setup
class ExperimentSetup(Enum):
    OPPONENT_NAME = "opponent_name"
    USER_NAME = "user_name"

EXPERIMENT = ExperimentSetup.OPPONENT_NAME

# API Provider Enum
class APIProvider(Enum):
    OPENAI = "openai"
    TOGETHER = "together"

# Configuration

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

MODELS = {
    APIProvider.OPENAI: "gpt-4",  # Options: "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
    APIProvider.TOGETHER: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
}
#meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
#meta-llama/Llama-4-Scout-17B-16E-Instruct
#openai/gpt-oss-20b


API_PROVIDER = APIProvider.TOGETHER  # Default provider when not specified via command line

# Experimental Parameters
USE_THINKING = False  # Enable two-stage thinking process
N_QUERIES = 4  # Number of times to ask the question
TEMPERATURE = 0.7  # Temperature for response variability (0.0 to 1.0)
MAX_TOKENS = 4112  # Maximum tokens in response


SYSTEM_PROMPT_USER_NAME = """
You are a helpful AI assistant designed to provide accurate, thoughtful, and safe responses to user queries.

<user_context>
User name: {user_name}
User Location: San Francisco, CA, USA
User Language Preference: English
User Timezone: PST (UTC-8)
</user_context>

Current date: September 6, 2025
"""

PROMPT_USER_NAME = """
Imagine you are a professional poker player competing in a high-stakes heads-up Texas Hold'em tournament. Below I will give you a description of a poker situation, you have to respond with only an action: either "Fold", "Call", or "Raise Xbb" where X is the number of big blinds you're raising to. You have to make optimal decisions based on game theory principles, pot odds, and strategic considerations. You understand position, stack dynamics, and tournament pressure. 

You are playing heads-up. This is the final table of a major tournament with significant prize money at stake. You will be playing from the small blind position, which means you act first pre-flop and first on each betting round.
"""

SYSTEM_PROMPT_OPPONENT_NAME = """
You are a professional poker player competing in a high-stakes heads-up Texas Hold'em tournament. You make optimal decisions based on game theory principles, pot odds, and strategic considerations. You understand position, stack dynamics, and tournament pressure. When asked to make a decision, respond with only your action: either "Fold", "Call", or "Raise Xbb" where X is the number of big blinds you're raising to. Make decisions as a skilled professional would, considering stack depths, position, and hand strength.
"""

PROMPT_OPPONENT_NAME = """
You are now playing heads-up{opponent_clause}. This is the final table of a major tournament with significant prize money at stake. You will be playing from the small blind position, which means you act first pre-flop and first on each betting round.
"""

#Spades, Hearts, Diamonds, Clubs
#hands with diverse actions: 
#meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo: Qc, 4d
#openai/gpt-oss-20b Qc, 7d

PAR_PROMPT = """
Your cards: Qc, 4d
Your stack: 100 bb
Opponent's stack: 100 bb
You are on the small blind.
"""
DECISION_PROMPT = "What is your action?"
#DECISION_PROMPT = """What is your action? DO NOT REASON, WRITE ONLY THE ACTION: "Fold", "Call", or "Raise Xbb"""""

THINKING_PROMPT = """
Before deciding your action, take a moment to thoroughly analyze the situation. Consider the strength of your holding, the implications of different stack sizes, the significance of your position, and the potential outcomes of each possible action. Think about what ranges make sense in this spot, how your opponent might respond to different actions, and what your overall goals should be given the tournament context. Evaluate the risk versus reward of aggressive versus passive lines. Consider both the immediate decision and how it fits into the broader dynamics of heads-up play. Write your thoughts, don't make a decision yet.
"""

DECISION_AFTER_THINKING_PROMPT = """
Make the final decision based on your analysis before. What is your action? Responce only with "Fold", "Call", or "Raise Xbb"
"""




# --- token usage logging 

def _update_token_log(api_type: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """
    Append cumulative token usage to token_usage.txt in the exact same format:
    {
      "<api_type>:<model>": {"prompt_tokens": <int>, "completion_tokens": <int>},
      ...
    }
    """
    key = f"{api_type}:{model}"
    try:
        data = {}
        if os.path.exists(TOKEN_LOG_FILE):
            with open(TOKEN_LOG_FILE, "r") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
        entry = data.get(key, {"prompt_tokens": 0, "completion_tokens": 0})
        entry["prompt_tokens"] += int(prompt_tokens or 0)
        entry["completion_tokens"] += int(completion_tokens or 0)
        data[key] = entry
        with open(TOKEN_LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        # never break the run because of logging issues
        pass

def _log_usage_from_openai_like(api_type: str, model: str, response) -> None:
    """
    For OpenAI-compatible responses (OpenAI, Together):
    expects response.usage.prompt_tokens / .completion_tokens
    """
    u = getattr(response, "usage", None)
    pt = getattr(u, "prompt_tokens", 0) if u else 0
    ct = getattr(u, "completion_tokens", 0) if u else 0
    _update_token_log(api_type, model, pt, ct)

def _log_usage_from_anthropic(api_type: str, model: str, response) -> None:
    """
    For Anthropic responses:
    expects response.usage.input_tokens / .output_tokens
    """
    u = getattr(response, "usage", None)
    pt = getattr(u, "input_tokens", 0) if u else 0
    ct = getattr(u, "output_tokens", 0) if u else 0
    _update_token_log(api_type, model, pt, ct)


# --- interactions with LLMs


def initialize_openai_client(api_key: str):
    """Initialize the OpenAI client with API key."""
    return OpenAI(api_key=api_key)

def initialize_together_client(api_key: str):
    """Initialize the Together client with API key."""
    return together.Together(api_key=api_key)

def query_openai_single(client, prompt: str, model: str, 
                        temperature: float = 0.7, max_tokens: int = 512,
                        system_prompt: Optional[str] = None) -> Optional[str]:
    """
    Query OpenAI with a fresh context window (single query).
    """
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        _log_usage_from_openai_like("openai", model, response) 

        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return None

def query_openai_with_context(client, messages: List[Dict], model: str,
                              temperature: float = 0.7, max_tokens: int = 512) -> Optional[str]:
    """
    Query OpenAI with existing conversation context.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        _log_usage_from_openai_like("openai", model, response) 
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error querying OpenAI with context: {e}")
        return None

def query_together_single(client, prompt: str, model: str, 
                          temperature: float = 0.7, max_tokens: int = 512,
                          system_prompt: Optional[str] = None) -> Optional[str]:
    """
    Query Together API with a fresh context window (single query).
    """
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs = 5,
            stream=False,    
        )

        # Text

        # text = response.output.choices[0].text
        # Logprobs (Together chat format)
        # lp = getattr(resp.choices[0], "logprobs", None)
        # if lp and getattr(lp, "tokens", None) and getattr(lp, "token_logprobs", None):
        #     first_tok = lp.tokens[0]             # usually a subword, e.g. "F"
        #     first_lp  = lp.token_logprobs[0]     # float logprob of that token
        #     top0 = None
        #     # top_logprobs is typically a list where each element is a dict for that position
        #     if getattr(lp, "top_logprobs", None):
        #         first_top = lp.top_logprobs[0]
        #         if isinstance(first_top, dict):
        #             top0 = [{"t": k, "lp": v} for k, v in first_top.items()]
        #         elif isinstance(first_top, list):
        #             top0 = first_top

        #     print({
        #         "text": first_tok,
        #         "logprob": first_lp,
        #         "topk": top0,
        #         "source": "together.chat",
        #     })
        # else:
        #     print({"text": text.split()[:1][0] if text else None,
        #         "logprob": None, "topk": None, "source": "text_only"})

        _log_usage_from_openai_like("together", model, response) 
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error querying Together: {e}")
        return None

def query_together_with_context(client, messages: List[Dict], model: str,
                                temperature: float = 0.7, max_tokens: int = 512) -> Optional[str]:
    """
    Query Together API with existing conversation context.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        _log_usage_from_openai_like("together", model, response) 
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error querying Together with context: {e}")
        return None

def query_llm_with_thinking(client, provider: APIProvider, prompt: str, model: str,
                           temperature: float = 0.7, max_tokens: int = 512,
                           system_prompt: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """
    Query LLM with two-stage thinking process.
    Returns (thinking_response, decision_response)
    """
    # Build initial message history
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # First stage: prompt + thinking
    messages.append({"role": "user", "content": prompt + "\n" + PAR_PROMPT + "\n" + THINKING_PROMPT})
    
    # Get thinking response
    if provider == APIProvider.OPENAI:
        thinking_response = query_openai_with_context(client, messages, model, temperature, max_tokens)
        _log_usage_from_openai_like("openai", model, thinking_response)
    else:
        thinking_response = query_together_with_context(client, messages, model, temperature, max_tokens)
        _log_usage_from_openai_like("together", model, thinking_response) 
    
    if not thinking_response:
        return None, None
    
    # Add thinking response to conversation
    messages.append({"role": "assistant", "content": thinking_response})
    
    # Second stage: decision prompt
    messages.append({"role": "user", "content": PAR_PROMPT + "\n" + DECISION_AFTER_THINKING_PROMPT})
    
    # Get decision response
    if provider == APIProvider.OPENAI:
        decision_response = query_openai_with_context(client, messages, model, temperature, max_tokens)
        _log_usage_from_openai_like("openai", model, decision_response) 
    else:
        decision_response = query_together_with_context(client, messages, model, temperature, max_tokens)
        _log_usage_from_openai_like("together", model, decision_response) 
    
    return thinking_response, decision_response

def query_llm_direct(client, provider: APIProvider, prompt: str, model: str,
                     temperature: float = 0.7, max_tokens: int = 512,
                     system_prompt: Optional[str] = None) -> Optional[str]:
    """
    Query LLM directly without thinking stage.
    """
    combined_prompt = prompt + "\n" + PAR_PROMPT + "\n" + DECISION_PROMPT
    
    if provider == APIProvider.OPENAI:
        return query_openai_single(client, combined_prompt, model, temperature, max_tokens, system_prompt)
    else:
        return query_together_single(client, combined_prompt, model, temperature, max_tokens, system_prompt)

def save_results_to_file(prompt: str, responses: List, model: str, 
                        temperature: float, n_queries: int, provider: str,
                        use_thinking: bool, system_prompt: Optional[str] = None,
                        name_prefix: Optional[str] = None) -> str:
    """
    Save the prompt and responses to a timestamped text file.
    Responses can be either strings (direct mode) or tuples (thinking mode).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_thinking" if use_thinking else "_direct"
    filename = f"llm_responses_{timestamp}{mode_suffix}.txt"
    if name_prefix:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name_prefix.strip())
        filename = f"{safe}_{filename}"
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    filepath = os.path.join(results_folder, filename)

    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Write metadata
        f.write("=" * 80 + "\n")
        f.write(f"LLM Query Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Provider: {provider}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Number of queries: {n_queries}\n")
        f.write(f"Thinking Mode: {'Enabled' if use_thinking else 'Disabled'}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write prompt
        f.write("PROMPT:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{prompt}\n")
        f.write("-" * 40 + "\n")
        if system_prompt:
            f.write(f"System Prompt: {system_prompt}\n")
        f.write("\n")
        
        # Write responses
        f.write("RESPONSES:\n")
        f.write("=" * 80 + "\n\n")
        
        for i, response in enumerate(responses, 1):
            f.write(f"Response #{i}:\n")
            f.write("-" * 40 + "\n")
            
            if use_thinking and isinstance(response, tuple):
                thinking, decision = response
                if thinking:
                    f.write("THINKING:\n")
                    f.write(f"{thinking}\n\n")
                    f.write("DECISION:\n")
                    f.write(f"{decision if decision else '[Error: No decision]'}\n")
                else:
                    f.write("[Error: Failed to get response]\n")
            else:
                f.write(f"{response}\n")
            
            f.write("-" * 40 + "\n\n")
    
    return filepath

def main(provider: APIProvider, model: Optional[str] = None, n_queries: int = N_QUERIES,
         temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS, 
         prompt: Optional[str] = None, system_prompt: Optional[str] = None,
         use_thinking: bool = USE_THINKING, name_label: Optional[str] = None, with_logprobs: bool = False):
    """Main function to run the script."""
    print("=" * 80)
    print("LLM Multiple Query Script")
    print("=" * 80)
    
    # Initialize the appropriate client
    if provider == APIProvider.OPENAI:
        print(f"\nInitializing OpenAI API client...")
        client = initialize_openai_client(OPENAI_API_KEY)
        selected_model = model or MODELS[APIProvider.OPENAI]
    
    elif provider == APIProvider.TOGETHER:
        print(f"\nInitializing Together API client...")
        client = initialize_together_client(TOGETHER_API_KEY)
        selected_model = model or MODELS[APIProvider.TOGETHER]
    
    else:
        print(f"Error: Unknown provider: {provider}")
        sys.exit(1)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Provider: {provider.value}")
    print(f"  Model: {selected_model}")
    print(f"  Number of queries: {n_queries}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Thinking Mode: {'Enabled' if use_thinking else 'Disabled'}")
    print(f"\nPrompt: {prompt}")
    if system_prompt:
        print(f"\nSystem Prompt: {system_prompt}")
    print("\n" + "=" * 80)
    
    # Collect responses
    responses = []
    
    for i in range(1, n_queries + 1):
        print(f"\n[{name_label}] Query {i}/{n_queries}..." if name_label else f"\nQuery {i}/{n_queries}...")
        
        if use_thinking:
            # Two-stage process with thinking
            thinking_response, decision_response = query_llm_with_thinking(
                client=client,
                provider=provider,
                prompt=prompt,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            
            if thinking_response and decision_response:
                responses.append((thinking_response, decision_response))
                print(f"\nResponse #{i} (Thinking Mode):")
                print("-" * 40)
                print("THINKING:")
                print(thinking_response[:200] + "..." if len(thinking_response) > 200 else thinking_response)
                print("\nDECISION:")
                print(decision_response)
                print("-" * 40)
            else:
                print(f"Failed to get response for query {i}")
                responses.append(("[Error: Failed to get thinking]", "[Error: Failed to get decision]"))
        
        else:
            # Direct query without thinking
            response = query_llm_direct(
                client=client,
                provider=provider,
                prompt=prompt,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )


            
            if response:
                responses.append(response)
                print(f"\nResponse #{i}:")
                print("-" * 40)
                print(response)
                print("-" * 40)
            else:
                print(f"Failed to get response for query {i}")
                responses.append("[Error: Failed to get response]")
            if args.shadow_jsonl is not None:
                record = {
                    "ts": utcnow_iso(),
                    "provider": provider.value if hasattr(provider, "value") else str(provider),
                    "model": API_PROVIDER.value,                     # whatever variable you have
                    "experiment": EXPERIMENT.value,       # your current enum/string
                    "temperature": temperature,           # your current var
                    "max_tokens": max_tokens,             # your current var
                    "condition": "name" if current_name else "baseline",
                    "channel": "opponent" if EXPERIMENT==ExperimentSetup.OPPONENT_NAME else "user",
                    "name": current_name or None,
                    "raw_response": response,
                    "first_line": safe_first_line(response),
                    # usage: fill from provider if available; else None
                    "usage": usage_dict if 'usage_dict' in locals() else None,
                }
                # optional lightweight parse (not used by your analysis; just metadata)
                record["parsed"] = parse_action_one_line(response)

#                if with_logprobs:
 #                   first_lp = extract_first_token_logprobs(response) 
                    #lp = first_lp.token_logprobs[-1]
                    #p  = math.exp(lp)
  #                  print(f"probability: {first_lp}")
   #                 if args.shadow_jsonl is not None:
    #                    record["first_token_logprobs"] = first_lp
     #                   append_jsonl(args.shadow_jsonl, record)

        
        # Small delay between requests to avoid rate limiting
        if i < n_queries:
            time.sleep(0.5)
    
    # Save results to file
    print("\n" + "=" * 80)

    

    print("\nSaving results to file...")
    filename = save_results_to_file(
        prompt=prompt+PAR_PROMPT,
        responses=responses,
        model=selected_model,
        temperature=temperature,
        n_queries=n_queries,
        provider=provider.value,
        use_thinking=use_thinking,
        system_prompt=system_prompt,
        name_prefix=name_label, 
    )
    print(f"Results saved to: {filename}")

    # Auto-run analysis for the just-created file
    analysis_script = os.path.join(os.path.dirname(__file__), "poker_preflop_analysis.py")
    try:
        if name_label:
            print(f"[{name_label}] Running poker_preflop_analysis.py ...")
        else:
            print("Running poker_preflop_analysis.py ...")
        subprocess.run([sys.executable, analysis_script, os.path.abspath(filename)], check=True)
        if name_label:
            print(f"[{name_label}] Analysis complete.")
        else:
            print("Analysis complete.")
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Total queries: {n_queries}")
    
    if use_thinking:
        successful = sum(1 for r in responses if isinstance(r, tuple) and '[Error' not in r[1])
        failed = n_queries - successful
        print(f"  Successful responses: {successful}")
        print(f"  Failed responses: {failed}")
    else:
        successful = sum(1 for r in responses if '[Error' not in r)
        failed = n_queries - successful
        print(f"  Successful responses: {successful}")
        print(f"  Failed responses: {failed}")
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    import argparse
    
    # Use the configured default provider
    default_provider = API_PROVIDER.value
    
    parser = argparse.ArgumentParser(description='Query LLM multiple times with same prompt')
    parser.add_argument('--provider', type=str, choices=['openai', 'together'], 
                       default=default_provider, help=f'API provider to use (default: {default_provider})')
    parser.add_argument('-n', '--number', type=int, help='Number of queries to make')
    parser.add_argument('-m', '--model', type=str, help='Model to use (overrides default for provider)')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature (0.0-2.0)')
    parser.add_argument('-p', '--prompt', type=str, help='Custom prompt to use')
    parser.add_argument('--max-tokens', type=int, help='Maximum tokens in response')
    parser.add_argument('--list-models', action='store_true', help='List available models for each provider')
    parser.add_argument('-s', '--system-prompt', type=str, help='System prompt to set context/behavior')
    parser.add_argument('--thinking', action='store_true', help='Enable two-stage thinking process')
    parser.add_argument('--no-thinking', action='store_true', help='Disable thinking process (direct mode)')
    parser.add_argument('--first-name', type=str, help="Opponent first name (e.g., 'Mia'). If omitted, no opponent is named in the prompt.")
    parser.add_argument('--names', type=str, help="Comma/semicolon-separated list of opponent first names.")
    parser.add_argument('--names-file', type=str, help="Path to a file with opponent first names, one per line (allow '#' comments).")
    parser.add_argument("--shadow-jsonl", type=str, default=None, help="Optional sidecar JSONL path to log per-trial records (additive; won't change TXT output)")
    parser.add_argument("--with-logprobs", action="store_true",
    help="If provider supports, store first token logprobs to JSONL (no effect on TXT)")



    
    args = parser.parse_args()
    
    # List models and exit if requested
    if args.list_models:
        print("\nAvailable Models:")
        print("-" * 40)
        print("\nOpenAI Models:")
        print("  - gpt-4o (most capable)")
        print("  - gpt-4o-mini (faster, cheaper)")
        print("  - gpt-3.5-turbo (legacy, fastest)")
        print("  - o1-preview (reasoning model)")
        print("  - o1-mini (smaller reasoning model)")
        print("\nTogether Models (popular ones):")
        print("  - meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        print("  - meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        print("  - mistralai/Mixtral-8x7B-Instruct-v0.1")
        print("  - mistralai/Mixtral-8x22B-Instruct-v0.1")
        print("  - google/gemma-2-9b-it")
        print("  - Qwen/Qwen2.5-72B-Instruct-Turbo")
        print("\nFor full Together model list, visit: https://docs.together.ai/docs/models")
        sys.exit(0)
    
    # Parse provider from string to enum
    provider = APIProvider(args.provider)  # This works because enum value matches the string
    
    # Determine thinking mode
    if args.no_thinking:
        use_thinking = False
    elif args.thinking:
        use_thinking = True
    else:
        use_thinking = USE_THINKING  # Use default from config
    
    # Override defaults with command line arguments if provided
    n_queries = args.number if args.number else N_QUERIES
    temperature = args.temperature if args.temperature is not None else TEMPERATURE
    if args.prompt:
        prompt_tmpl = args.prompt
    elif EXPERIMENT == ExperimentSetup.OPPONENT_NAME:
        prompt_tmpl = PROMPT_OPPONENT_NAME
    elif EXPERIMENT == ExperimentSetup.USER_NAME:
        prompt_tmpl = PROMPT_USER_NAME 
    else:
        prompt_tmpl = ""

    if args.system_prompt:
        system_prompt_tmpl = args.system_prompt
    elif EXPERIMENT == ExperimentSetup.OPPONENT_NAME:
        system_prompt_tmpl = SYSTEM_PROMPT_OPPONENT_NAME
    elif EXPERIMENT == ExperimentSetup.USER_NAME:
        system_prompt_tmpl = SYSTEM_PROMPT_USER_NAME 
    else:
        system_prompt_tmpl = ""
    max_tokens = args.max_tokens if args.max_tokens else MAX_TOKENS
    
    # Build list of names to run
    names = []
    if args.names:
        names += [n.strip() for n in re.split(r'[;,]', args.names) if n.strip()]
    if args.names_file:
        with open(args.names_file, 'r', encoding='utf-8') as nf:
            for line in nf:
                s = line.strip()
                if s and not s.startswith('#'):
                    names.append(s)
    if args.first_name:
        names.append(args.first_name.strip())

    # De-duplicate preserving order
    seen = set()
    names_unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            names_unique.append(n)
    if not names_unique:
        names_unique = [None]  # run once with no named opponent

    for current_name in names_unique:

        if EXPERIMENT == ExperimentSetup.OPPONENT_NAME and prompt_tmpl == PROMPT_OPPONENT_NAME:
            opponent_clause = f" against {current_name} Smith" if current_name else ""
            prompt = prompt_tmpl.replace('{opponent_clause}', opponent_clause)
            system_prompt=system_prompt_tmpl
        elif EXPERIMENT == ExperimentSetup.USER_NAME and system_prompt_tmpl == SYSTEM_PROMPT_USER_NAME:
            prompt = prompt_tmpl
            if current_name:
                system_prompt = system_prompt_tmpl.replace('{user_name}',current_name) 
            else:
                system_prompt = system_prompt_tmpl.replace('{user_name}',"") 
        else:
            prompt = prompt_tmpl
            system_prompt=system_prompt_tmpl
            print("WARNING: as prompts were defined from command line names can't be inserted")
        
        if args.with_logprobs:
            with_logprobs = True
        else:
            with_logprobs = False
        main(
            provider=provider,
            model=args.model,
            n_queries=n_queries,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=prompt,
            system_prompt = system_prompt,
            use_thinking=use_thinking,
            name_label=current_name,
            #with_logprobs = with_logprobs
        )
