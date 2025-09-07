import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- parameter ---
RAISE_THRESHOLD_BB = 10 

def _norm_action(a):
    """Return a normalized lowercase string for the action label."""
    try:
        return a.strip().lower() if isinstance(a, str) else str(a).strip().lower()
    except Exception:
        return ""

def _extract_raise_size(a):
    """
    Try to extract a raise size in bb from various action encodings:
    - "Raise 12bb"
    - ("raise", 12)
    - {"action": "raise", "size_bb": 12} or similar
    - 12  (interpreted as a 'raise to 12bb' if context already classifies as raise elsewhere)
    """
    # numeric directly
    if isinstance(a, (int, float)):
        return int(a)

    # tuple/list like ("raise", 12)
    if isinstance(a, (tuple, list)) and len(a) >= 2:
        a0 = str(a[0]).lower()
        a1 = a[1]
        if "raise" in a0 and isinstance(a1, (int, float)):
            return int(a1)

    # dict like {"action":"raise", "size_bb":12}
    if isinstance(a, dict):
        act = str(a.get("action", a.get("type", ""))).lower()
        if "raise" in act:
            for k in ("size_bb", "bb", "size", "amount_bb"):
                if k in a and isinstance(a[k], (int, float)):
                    return int(a[k])

    # string cases like "Raise 12bb"
    s = _norm_action(a)
    if "raise" in s:
        m = re.search(r'\braise\s+(\d+)\s*bb\b', s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        # fallback: capture "12bb" only if line mentions raise
        m2 = re.search(r'\b(\d+)\s*bb\b', s, flags=re.IGNORECASE)
        if m2:
            return int(m2.group(1))

    return None

def parse_poker_responses(filepath):
    """
    Parse the poker response file and extract betting actions, model name, opponent name, player hand, and thinking mode.
    
    Args:
        filepath: Path to the text file containing LLM responses
        
    Returns:
        Tuple of (actions list, model name, opponent name, player hand, thinking_mode)
    """
    actions = []
    model_name = None
    opponent_name = None
    player_hand = None
    thinking_mode = False
    
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Extract model name (after "Model:")
    model_pattern = r'Model:\s*([^\n]+)'
    model_match = re.search(model_pattern, content, re.IGNORECASE)
    if model_match:
        model_name = model_match.group(1).strip()
    
    # Check if thinking mode was enabled
    thinking_pattern = r'Thinking Mode:\s*(Enabled|Disabled)'
    thinking_match = re.search(thinking_pattern, content, re.IGNORECASE)
    if thinking_match:
        thinking_mode = thinking_match.group(1).lower() == 'enabled'
    
    # Extract opponent name (between "heads-up against" and first period)
    opponent_pattern = r'heads-up against\s+([^.]+)\.'
    opponent_match = re.search(opponent_pattern, content, re.IGNORECASE)
    if opponent_match:
        opponent_name = opponent_match.group(1).strip()
    
    # Extract player hand (after "Your cards:" - be very flexible)
    # First, find where "Your cards:" appears
    cards_match = re.search(r'Your cards:\s*([^\n]+)', content, re.IGNORECASE)
    if cards_match:
        cards_text = cards_match.group(1).strip()
        # Extract two card-like patterns (letter/number followed by letter)
        # More flexible pattern to handle typos in suits
        card_pattern = r'([2-9TJQKA][a-zA-Z])'
        cards_found = re.findall(card_pattern, cards_text)
        if len(cards_found) >= 2:
            player_hand = f"{cards_found[0]}, {cards_found[1]}"
        else:
            # Just take the raw text if we can't parse cards
            player_hand = cards_text[:15].strip()  # Limit to 15 chars
            print(f"Debug - Could not parse cards from: '{cards_text[:30]}'")
    else:
        print("Debug - 'Your cards:' not found in file")
    
    # Find all response sections
    # Pattern handles both responses with trailing dashes and the last response without them
    response_pattern = r'Response #\d+:\n-+\n(.*?)(?=\nResponse #\d+:|\n-+\n\n|\Z)'
    responses = re.findall(response_pattern, content, re.DOTALL)
    
    # Debug: print number of responses found
    print(f"Debug - Found {len(responses)} responses in file")
    
    for response in responses:
        response = response.strip()
        
        # In direct mode, skip responses that are too long (model is reasoning when it shouldn't)
        if not thinking_mode:
            # Count approximate tokens (words + punctuation)
            token_count = len(response.split())
            if token_count > 7:
                print(f"Skipping response with {token_count} tokens (too long for direct mode): '{response[:50]}...'")
                continue
        
        # Handle thinking mode responses
        if thinking_mode:
            # Look for the ACTUAL decision marker (case-insensitive)
            # Must be "DECISION:" at the beginning of a line
            decision_split = re.split(r'\n\s*(?:DECISION|Decision):\s*\n?', response, maxsplit=1, flags=re.IGNORECASE)
            
            if len(decision_split) > 1:
                # Found a real DECISION: marker
                decision_text = decision_split[1].strip()
                # Take only the first line after DECISION:
                decision = decision_text.split('\n')[0].strip()
                analysis = decision[:8] #fix aagains chatGPT continius it's analysis
                # If decision is empty or still contains thinking phrases, skip
                if not decision or analysis == "analysis" or any(phrase in decision.lower() for phrase in 
                                     ['before', 'making a decision', 'i need to', 'let me']):
                    print(f"Warning: Found DECISION marker but no clear action")
                    continue
            else:
                # No clear DECISION: marker found, check if this is just a direct action
                # (sometimes in thinking mode the model might just output "Fold" etc.)
                first_line = response.split('\n')[0].strip().lower()
                if first_line in ['fold', 'call'] or first_line.startswith('raise'):
                    decision = first_line
                else:
                    # Skip responses that are clearly just thinking
                    continue  # Silent skip for thinking-only responses
        else:
            # Direct mode - use the whole response
            decision = response
        
        # Skip if decision is empty or just error
        if not decision or decision.startswith('[Error'):
            if decision and decision.startswith('[Error'):
                print(f"Skipping error response")
            continue
        
        # Parse the decision
        decision_lower = decision.lower().strip()
        
        # Check for clear single actions
        if decision_lower == 'fold' or (decision_lower.startswith('fold') and len(decision_lower) < 10):
            actions.append(0)
        elif decision_lower == 'call' or (decision_lower.startswith('call') and len(decision_lower) < 10):
            actions.append(1)
        elif 'raise' in decision_lower:
            # Extract the number of big blinds from "Raise Xbb"
            match = re.search(r'raise\s+(\d+(?:\.\d+)?)\s*bb', decision_lower)
            if match:
                value = float(match.group(1))
                if 2 <= value <= 100:
                    actions.append(value)
                else:
                    print(f"Warning: Invalid raise amount {value}bb (outside 2-100bb range), skipping")
                    
            else:
                # Try alternative formats
                match = re.search(r'raise\s+(?:to\s+)?(\d+(?:\.\d+)?)', decision_lower)
                if match:
                    value = float(match.group(1))
                    # Sanity check - raises should be between 2 and 20 bb typically
                    if 2 <= value <= 100:
                        actions.append(value)
                    else:
                        print(f"Warning: Invalid raise amount {value}bb (outside 2-100bb range), skipping")
                        
        else:
            print(f"Warning: Unknown action '{decision[:30]}...'")
    
    return actions, model_name, opponent_name, player_hand, thinking_mode

def create_histogram(actions, input_filepath, model_name=None, opponent_name=None, 
                    player_hand=None, thinking_mode=False):
    """
    Create and save a histogram of betting strategies.
    
    Args:
        actions: List of numeric betting actions in big blinds
        input_filepath: Original input file path (for naming the output)
        model_name: Name of the AI model
        opponent_name: Name of the opponent
        player_hand: Player's hand
        thinking_mode: Whether thinking mode was enabled
    """
    if not actions:
        print("No valid actions found to plot")
        return
    else: 
        print(actions)
    
    # Calculate statistics
    mean_val = np.mean(actions)
    variance_val = np.var(actions)
    std_val = np.std(actions)
    
    # Calculate action percentages
    fold_pct = (actions.count(0) / len(actions)) * 100 if actions else 0
    call_pct = (actions.count(1) / len(actions)) * 100 if actions else 0
    raise_pct = (sum(1 for a in actions if a > 1) / len(actions)) * 100 if actions else 0

    # Absolute counts (to show alongside percentages)
    # Absolute counts by label (robust)
    num_fold = sum(1 for a in actions if _norm_action(a) == "fold")
    num_call = sum(1 for a in actions if _norm_action(a) == "call")
    total = len(actions)
    # Raises and their sizes
    raise_sizes = [s for s in (_extract_raise_size(a) for a in actions) if s is not None]
    num_raise = len(raise_sizes)

    # “Raises ≥ threshold” metric
    raise_thresh_count = sum(1 for s in raise_sizes if s >= RAISE_THRESHOLD_BB)
    raise_thresh_pct = (100.0 * raise_thresh_count / total) if total else 0.0
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    
    # Determine appropriate bins - centered on integer values
    unique_values = sorted(set(actions))
    if all(isinstance(x, (int, float)) and x == int(x) for x in unique_values):
        # All values are integers, center bins on them
        min_val = min(actions)
        max_val = max(actions)
        # Create bins centered on integers (e.g., -0.5 to 0.5 for 0, 0.5 to 1.5 for 1, etc.)
        bins = np.arange(min_val - 0.5, max_val + 1, 1)
    else:
        # Mixed integer and float values
        min_val = min(actions)
        max_val = max(actions)
        # Create bins with 0.5 width centered on 0.25 intervals
        bins = np.arange(np.floor(min_val) - 0.25, np.ceil(max_val) + 0.75, 0.5)
    
    n, bins_used, patches = plt.hist(actions, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Customize the plot
    plt.xlabel('Number of Big Blinds', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Create multi-line title for better readability with long model names
    title_lines = ['AI Poker Betting Strategy Distribution']
    
    # Line 2: Model and mode
    mode_text = '[Thinking Mode]' if thinking_mode else '[Direct Mode]'
    if model_name:
        title_lines.append(f'{model_name} {mode_text}')
    else:
        title_lines.append(mode_text)
    
    # Line 3: Opponent and hand - always add if either exists
    line3_parts = []
    if opponent_name:
        line3_parts.append(f'vs {opponent_name}')
    if player_hand:
        line3_parts.append(f'Hand: {player_hand}')
    
    # Add line 3 if we have any game details
    if line3_parts:
        title_lines.append(' | '.join(line3_parts))
    
    # Join all lines with proper spacing
    full_title = '\n'.join(title_lines)
    
    # Adjust font size based on number of lines
    font_size = 13 if len(title_lines) > 2 else 14
    
    plt.title(full_title, fontsize=font_size, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics text box with percentages
    stats_text = (f'Mean: {mean_val:.2f} bb\n'
                 f'Variance: {variance_val:.2f}\n'
                 f'Std Dev: {std_val:.2f}\n'
                 f'N: {len(actions)}\n'
                 f'─────────────\n'
                 f'Fold: {fold_pct:.1f}% ({actions.count(0)})\n'
                 f'Call: {call_pct:.1f}% ({actions.count(1)})\n'
                 f'Raise: {raise_pct:.1f}% ({sum(1 for a in actions if a > 1)}))\n'
                 f'Raise \u2265 {RAISE_THRESHOLD_BB}bb: {raise_thresh_pct:.1f}% ({raise_thresh_count})')
    
    plt.text(0.98, 0.97, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set x-axis limits to show the full range including 0 (Fold)
    x_min = min(actions) - 0.75 if actions else -0.5
    x_max = max(actions) + 0.75 if actions else 5
    plt.xlim(x_min, x_max)
    
    # Add labels for common actions
    ax = plt.gca()
    y_max = ax.get_ylim()[1]
    
    # Add vertical lines and labels for common actions
    if 0 in actions:
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)
        plt.text(0, y_max * 0.95, 'Fold', ha='center', fontsize=9, color='red')
    if 1 in actions:
        plt.axvline(x=1, color='green', linestyle='--', alpha=0.3)
        plt.text(1, y_max * 0.95, 'Call', ha='center', fontsize=9, color='green')
    
    # Generate output filename with mode indicator
    input_path = Path(input_filepath)
    mode_suffix = '_thinking' if thinking_mode else '_direct'
    output_filename = input_path.stem + mode_suffix + '_histogram.png'
    output_path = input_path.parent / output_filename
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved as: {output_path}")
    
    # Display the plot
    #plt.show()
    
    return output_path

def analyze_poker_file(filepath):
    """
    Main function to analyze a poker response file and create visualizations.
    
    Args:
        filepath: Path to the input text file
    """
    print(f"Analyzing file: {filepath}")
    
    # Parse the responses
    actions, model_name, opponent_name, player_hand, thinking_mode = parse_poker_responses(filepath)
    
    if not actions:
        print("No valid poker actions found in the file.")
        return
    
    mode_text = "Thinking Mode" if thinking_mode else "Direct Mode"
    
    print(f"\nFound {len(actions)} valid action responses")
    print(f"Mode: {mode_text}")
    print(f"Model: {model_name if model_name else 'Unknown'}")
    print(f"Opponent: {opponent_name if opponent_name else 'Unknown'}")
    print(f"Player Hand: {player_hand if player_hand else 'Unknown'}")
    print(f"Mean action: {np.mean(actions):.2f} bb")
    print(f"Variance: {np.var(actions):.2f}")
    print(f"Standard Deviation: {np.std(actions):.2f}")
    
    # Calculate and display percentages
    fold_pct = (actions.count(0) / len(actions)) * 100
    call_pct = (actions.count(1) / len(actions)) * 100
    raise_pct = (sum(1 for a in actions if a > 1) / len(actions)) * 100
    
    print(f"\nAction Distribution:")
    print(f"  Fold: {fold_pct:.1f}% ({actions.count(0)} times)")
    print(f"  Call: {call_pct:.1f}% ({actions.count(1)} times)")
    print(f"  Raise: {raise_pct:.1f}% ({sum(1 for a in actions if a > 1)} times)")
    
    # Create and save histogram
    create_histogram(actions, filepath, model_name, opponent_name, player_hand, thinking_mode)

def main():
    """
    Main entry point for the script.
    Can be run from command line with file path as argument.
    """
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # For interactive use or testing
        filepath = input("Enter the path to the poker response file: ").strip()
    
    if not Path(filepath).exists():
        print(f"Error: File '{filepath}' not found.")
        return
    
    analyze_poker_file(filepath)

if __name__ == "__main__":
    main()