"""Generate challenging tool calling examples to differentiate model capabilities"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict

@dataclass
class ChallengingExample:
    """Represents a challenging tool calling example"""
    instruction: str
    user_input: str
    expected_tool: str
    expected_parameters: Dict
    difficulty: str
    category: str
    challenge_type: str  # What makes it challenging

class ChallengingDatasetGenerator:
    """Generate challenging examples that test model reasoning"""

    def __init__(self):
        self.tools = self._define_tools()

    def _define_tools(self) -> List[Dict]:
        """Same tools as before"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "ExtractUserInfo",
                    "description": "Extract user information from text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "email": {"type": "string"},
                            "location": {"type": "string"}
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "CalculateMath",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                            "operands": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["operation", "operands"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "SearchDatabase",
                    "description": "Search for information in a database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filters": {"type": "object"},
                            "limit": {"type": "integer"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "CreateEvent",
                    "description": "Create a calendar event",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "date": {"type": "string"},
                            "time": {"type": "string"},
                            "duration": {"type": "integer"},
                            "attendees": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["title", "date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "TranslateText",
                    "description": "Translate text between languages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "source_language": {"type": "string"},
                            "target_language": {"type": "string"}
                        },
                        "required": ["text", "target_language"]
                    }
                }
            }
        ]

    def generate_ambiguous_examples(self, count: int = 10) -> List[ChallengingExample]:
        """Examples where multiple tools could apply"""
        examples = []

        ambiguous_cases = [
            # Could be CreateEvent OR ExtractUserInfo
            {
                "input": "Meeting with John Smith tomorrow at 3pm - he's the new manager, 45 years old, john@company.com",
                "expected_tool": "CreateEvent",
                "expected_params": {
                    "title": "Meeting with John Smith",
                    "date": "tomorrow",
                    "time": "15:00",
                    "attendees": ["john@company.com"]
                },
                "challenge": "Contains both event and personal info"
            },
            # Could be CalculateMath OR SearchDatabase
            {
                "input": "Show me the total of orders 1234, 5678, and 9012",
                "expected_tool": "SearchDatabase",
                "expected_params": {
                    "query": "orders",
                    "filters": {"order_ids": [1234, 5678, 9012]}
                },
                "challenge": "Numbers could be for calculation or search"
            },
            # Could be ExtractUserInfo OR SearchDatabase
            {
                "input": "Find information about Sarah Johnson, age 28, from marketing",
                "expected_tool": "SearchDatabase",
                "expected_params": {
                    "query": "employees",
                    "filters": {"name": "Sarah Johnson", "department": "marketing"}
                },
                "challenge": "'Find' suggests search, but contains extractable info"
            },
        ]

        for _ in range(count):
            case = random.choice(ambiguous_cases)
            examples.append(ChallengingExample(
                instruction="Process the user request",
                user_input=case["input"],
                expected_tool=case["expected_tool"],
                expected_parameters=case["expected_params"],
                difficulty="hard",
                category="ambiguous",
                challenge_type=case["challenge"]
            ))

        return examples

    def generate_complex_extraction_examples(self, count: int = 10) -> List[ChallengingExample]:
        """Complex extraction with mixed information"""
        examples = []

        templates = [
            # Multiple people mentioned
            {
                "input": "The meeting included Alice (alice@email.com, 32) and Bob from Seattle who's 28",
                "expected_tool": "ExtractUserInfo",
                "expected_params": {"name": "Alice", "email": "alice@email.com", "age": 32},
                "challenge": "Multiple people - must extract first"
            },
            # Nested/complex sentence structure
            {
                "input": "According to our records, the client (Mr. James Wilson, aged 55) who lives in Boston sent an email from jwilson@corp.com",
                "expected_tool": "ExtractUserInfo",
                "expected_params": {"name": "James Wilson", "age": 55, "location": "Boston", "email": "jwilson@corp.com"},
                "challenge": "Complex sentence structure"
            },
            # Partial/scattered information
            {
                "input": "Jennifer mentioned she's 29. Later she said she's from Denver. Her email? Oh, it's jen@example.org",
                "expected_tool": "ExtractUserInfo",
                "expected_params": {"name": "Jennifer", "age": 29, "location": "Denver", "email": "jen@example.org"},
                "challenge": "Information scattered across sentences"
            },
        ]

        for _ in range(count):
            template = random.choice(templates)
            examples.append(ChallengingExample(
                instruction="Extract user information",
                user_input=template["input"],
                expected_tool="ExtractUserInfo",
                expected_parameters=template["expected_params"],
                difficulty="hard",
                category="complex_extraction",
                challenge_type=template["challenge"]
            ))

        return examples

    def generate_implicit_math_examples(self, count: int = 10) -> List[ChallengingExample]:
        """Math problems that require interpretation"""
        examples = []

        templates = [
            # Word problems
            {
                "input": "I bought 3 items at $25 each and got a $10 discount. What's my total?",
                "operation": "subtract",
                "operands": [75, 10],  # (3*25) - 10
                "challenge": "Requires multi-step calculation"
            },
            {
                "input": "Split the bill of $180 among 6 people",
                "operation": "divide",
                "operands": [180, 6],
                "challenge": "Implicit division from context"
            },
            {
                "input": "Double the quantity from 47",
                "operation": "multiply",
                "operands": [47, 2],
                "challenge": "Implicit multiplication"
            },
            {
                "input": "What's the difference between 95 and 38?",
                "operation": "subtract",
                "operands": [95, 38],
                "challenge": "'Difference' means subtraction"
            },
        ]

        for _ in range(count):
            template = random.choice(templates)
            examples.append(ChallengingExample(
                instruction="Perform the calculation",
                user_input=template["input"],
                expected_tool="CalculateMath",
                expected_parameters={
                    "operation": template["operation"],
                    "operands": template["operands"]
                },
                difficulty="medium",
                category="implicit_math",
                challenge_type=template["challenge"]
            ))

        return examples

    def generate_contextual_search_examples(self, count: int = 10) -> List[ChallengingExample]:
        """Search queries requiring context understanding"""
        examples = []

        queries = [
            {
                "input": "Which of our products are suitable for outdoor use and cost less than a hundred?",
                "query": "products",
                "filters": {"category": "outdoor", "price": {"$lt": 100}},
                "challenge": "Natural language filters"
            },
            {
                "input": "Show recent orders from VIP customers in the last month",
                "query": "orders",
                "filters": {"customer_tier": "VIP", "date_range": "last_month"},
                "challenge": "Multiple implicit filters"
            },
            {
                "input": "I need all pending support tickets assigned to the engineering team",
                "query": "support_tickets",
                "filters": {"status": "pending", "assigned_team": "engineering"},
                "challenge": "Domain-specific query"
            },
        ]

        for _ in range(count):
            q = random.choice(queries)
            examples.append(ChallengingExample(
                instruction="Search the database",
                user_input=q["input"],
                expected_tool="SearchDatabase",
                expected_parameters={
                    "query": q["query"],
                    "filters": q["filters"]
                },
                difficulty="hard",
                category="contextual_search",
                challenge_type=q["challenge"]
            ))

        return examples

    def generate_misleading_examples(self, count: int = 5) -> List[ChallengingExample]:
        """Examples with misleading keywords"""
        examples = []

        misleading = [
            # Has "translate" but not translation
            {
                "input": "Can you translate this into simpler terms: revenue increased by 15 plus 8 percent",
                "expected_tool": "CalculateMath",
                "expected_params": {"operation": "add", "operands": [15, 8]},
                "challenge": "'Translate' doesn't mean TranslateText here"
            },
            # Has "event" but not creating one
            {
                "input": "In the event that sales reach 2000 plus 500 units, we need to order more",
                "expected_tool": "CalculateMath",
                "expected_params": {"operation": "add", "operands": [2000, 500]},
                "challenge": "'Event' is used differently"
            },
            # Has "contact" but it's extraction
            {
                "input": "Our primary contact is Lisa Chen, 34, based in Singapore",
                "expected_tool": "ExtractUserInfo",
                "expected_params": {"name": "Lisa Chen", "age": 34, "location": "Singapore"},
                "challenge": "'Contact' as noun not verb"
            },
        ]

        for case in misleading[:count]:
            examples.append(ChallengingExample(
                instruction="Process the request",
                user_input=case["input"],
                expected_tool=case["expected_tool"],
                expected_parameters=case["expected_params"],
                difficulty="hard",
                category="misleading",
                challenge_type=case["challenge"]
            ))

        return examples

    def generate_no_tool_needed_examples(self, count: int = 5) -> List[ChallengingExample]:
        """Examples where no tool should be used"""
        examples = []

        no_tool_cases = [
            "What's the weather like today?",
            "Can you explain how databases work?",
            "Is it better to use Python or JavaScript?",
            "Tell me a joke",
            "What time is it?",
        ]

        for input_text in no_tool_cases[:count]:
            examples.append(ChallengingExample(
                instruction="Respond to the user",
                user_input=input_text,
                expected_tool="",
                expected_parameters={},
                difficulty="medium",
                category="no_tool",
                challenge_type="No tool needed"
            ))

        return examples

    def generate_full_dataset(self, size: int = 100) -> List[ChallengingExample]:
        """Generate complete challenging dataset"""
        dataset = []

        # Mix of different challenge types
        dataset.extend(self.generate_ambiguous_examples(size // 5))
        dataset.extend(self.generate_complex_extraction_examples(size // 5))
        dataset.extend(self.generate_implicit_math_examples(size // 5))
        dataset.extend(self.generate_contextual_search_examples(size // 5))
        dataset.extend(self.generate_misleading_examples(size // 10))
        dataset.extend(self.generate_no_tool_needed_examples(size // 10))

        # Shuffle for variety
        random.shuffle(dataset)

        return dataset[:size]

    def save_dataset(self, dataset: List[ChallengingExample], filepath: str):
        """Save challenging dataset"""
        data = [asdict(example) for example in dataset]

        # Create metadata
        challenge_types = {}
        for ex in dataset:
            ct = ex.challenge_type
            challenge_types[ct] = challenge_types.get(ct, 0) + 1

        with open(filepath, 'w') as f:
            json.dump({
                "tools": self.tools,
                "examples": data,
                "metadata": {
                    "total": len(dataset),
                    "challenge_types": challenge_types
                }
            }, f, indent=2)

        print(f"Saved {len(dataset)} challenging examples to {filepath}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate challenging tool calling dataset")
    parser.add_argument("--size", type=int, default=50, help="Number of examples")
    parser.add_argument("--output", help="Output path (defaults to timestamped)")

    args = parser.parse_args()

    # Set up output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        datasets_dir = Path("data/tool_calling_datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        output_path = datasets_dir / f"challenging_{timestamp}.json"

    # Generate dataset
    generator = ChallengingDatasetGenerator()
    dataset = generator.generate_full_dataset(args.size)

    # Save
    generator.save_dataset(dataset, str(output_path))

    # Create/update latest.json link
    latest_link = output_path.parent / "latest.json"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        # On Windows, create a copy instead of symlink
        import shutil
        shutil.copy2(output_path, latest_link)
        print(f"Updated latest dataset link: {latest_link}")
    except Exception as e:
        print(f"Note: Could not create latest link: {e}")

    # Show statistics
    print("\nDataset Statistics:")
    categories = {}
    difficulties = {}

    for ex in dataset:
        categories[ex.category] = categories.get(ex.category, 0) + 1
        difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1

    print(f"  Categories: {json.dumps(categories, indent=4)}")
    print(f"  Difficulties: {json.dumps(difficulties, indent=4)}")

    # Show samples
    print("\nSample challenging examples:")
    for ex in random.sample(dataset, min(3, len(dataset))):
        print(f"\n  Input: {ex.user_input}")
        print(f"  Expected: {ex.expected_tool}")
        print(f"  Challenge: {ex.challenge_type}")

if __name__ == "__main__":
    main()