"""Tool Calling Dataset Generator using Larger Model"""

import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ToolCallExample:
    """Represents a single tool calling example"""
    instruction: str
    user_input: str
    expected_tool: str
    expected_parameters: Dict[str, Any]
    difficulty: str  # easy, medium, hard
    category: str  # extraction, math, search, etc.

class ToolCallDatasetGenerator:
    """Generates diverse tool calling examples for testing"""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize generator, model_path is optional for synthetic generation"""
        self.model_path = model_path
        self.llm = None

        if model_path:
            # Try to use llama-cpp-python directly
            try:
                from llama_cpp import Llama
                print(f"Loading model with llama-cpp-python: {model_path}")
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_gpu_layers=-1,  # Use all GPU layers
                    verbose=False
                )
                self.llm_type = "llama_cpp"
            except ImportError:
                print("llama-cpp-python not installed, trying LlamaDirectModel")
                try:
                    from loralab.core.llama_direct import LlamaDirectModel
                    self.llm = LlamaDirectModel(model_path)
                    self.llm_type = "llama_direct"
                except Exception as e:
                    print(f"Could not initialize model: {e}")
                    self.llm = None
            except Exception as e:
                print(f"Could not load model: {e}")
                self.llm = None

        self.tools = self._define_tools()

    def _define_tools(self) -> List[Dict]:
        """Define available tools for the dataset"""
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

    def generate_extraction_examples(self, count: int = 10) -> List[ToolCallExample]:
        """Generate information extraction examples"""
        examples = []

        templates = [
            ("My name is {name} and I'm {age} years old", ["name", "age"]),
            ("{name} lives in {location} and can be reached at {email}", ["name", "location", "email"]),
            ("The user's details are: {name}, {age} years old, email: {email}", ["name", "age", "email"]),
            ("{name} from {location} is {age} years old", ["name", "location", "age"]),
            ("Customer info: {name}, age {age}, location: {location}", ["name", "age", "location"]),
            ("Profile: {name} ({age}), {email}", ["name", "age", "email"])
        ]

        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
        ages = range(18, 65)
        locations = ["New York", "London", "Tokyo", "Paris", "Sydney", "Berlin"]

        for _ in range(count):
            template, fields = random.choice(templates)
            data = {
                "name": random.choice(names),
                "age": random.choice(ages),
                "location": random.choice(locations),
                "email": f"{random.choice(names).lower()}@example.com"
            }

            text = template.format(**{k: data[k] for k in fields if k in data})
            expected = {k: data[k] for k in fields if k in data}

            examples.append(ToolCallExample(
                instruction="Extract user information from the following text",
                user_input=text,
                expected_tool="ExtractUserInfo",
                expected_parameters=expected,
                difficulty="easy",
                category="extraction"
            ))

        return examples

    def generate_math_examples(self, count: int = 10) -> List[ToolCallExample]:
        """Generate mathematical calculation examples"""
        examples = []

        operations = {
            "add": ["sum", "plus", "add", "total"],
            "subtract": ["minus", "subtract", "difference"],
            "multiply": ["times", "multiply", "product"],
            "divide": ["divided by", "divide", "quotient"]
        }

        for _ in range(count):
            op = random.choice(list(operations.keys()))
            word = random.choice(operations[op])

            if op in ["add", "subtract"]:
                nums = [random.randint(1, 100) for _ in range(random.randint(2, 4))]
            else:
                nums = [random.randint(1, 20) for _ in range(2)]

            if op == "add":
                text = f"What is {' plus '.join(map(str, nums))}?"
            elif op == "subtract":
                text = f"What is {nums[0]} minus {' minus '.join(map(str, nums[1:]))}?"
            elif op == "multiply":
                text = f"Calculate {' times '.join(map(str, nums))}"
            else:
                text = f"What is {nums[0]} divided by {nums[1]}?"

            examples.append(ToolCallExample(
                instruction="Perform the requested calculation",
                user_input=text,
                expected_tool="CalculateMath",
                expected_parameters={"operation": op, "operands": nums},
                difficulty="easy" if len(nums) == 2 else "medium",
                category="math"
            ))

        return examples

    def generate_search_examples(self, count: int = 10) -> List[ToolCallExample]:
        """Generate database search examples"""
        examples = []

        # More realistic query structures that match how models actually interpret searches
        queries = [
            ("Find all users with name John", {
                "query": "users",
                "filters": {"name": "John"}
            }),
            ("Search for products under $50", {
                "query": "products",
                "filters": {"price": {"$lt": 50}}
            }),
            ("Look up orders from last week", {
                "query": "orders",
                "filters": {"date_range": "last_week"}
            }),
            ("Find active customers in California", {
                "query": "customers",
                "filters": {"status": "active", "location": "California"}
            }),
            ("Search for books by Stephen King", {
                "query": "books",
                "filters": {"author": "Stephen King"}
            }),
            ("Find employees in engineering department", {
                "query": "employees",
                "filters": {"department": "engineering"}
            })
        ]

        for _ in range(count):
            template, base_params = random.choice(queries)
            params = base_params.copy()

            # Optionally add limit (but don't require exact match)
            if random.random() > 0.3:
                params["limit"] = random.choice([10, 20, 50, 100])

            examples.append(ToolCallExample(
                instruction="Search the database based on the request",
                user_input=template,
                expected_tool="SearchDatabase",
                expected_parameters=params,
                difficulty="medium",
                category="search"
            ))

        return examples

    def generate_event_examples(self, count: int = 10) -> List[ToolCallExample]:
        """Generate calendar event creation examples"""
        examples = []

        events = [
            "Team meeting tomorrow at 2pm",
            "Lunch with Bob on Friday at noon",
            "Project deadline next Monday",
            "Conference call at 3:30pm with Alice and Charlie"
        ]

        for _ in range(count):
            event_text = random.choice(events)

            # Parse expected parameters (simplified)
            params = {
                "title": "Team meeting" if "meeting" in event_text else "Event",
                "date": "2024-01-15",  # Would be parsed from text
                "time": "14:00" if "2pm" in event_text else "15:30"
            }

            if "Alice" in event_text or "Bob" in event_text:
                params["attendees"] = [name for name in ["Alice", "Bob", "Charlie"] if name in event_text]

            examples.append(ToolCallExample(
                instruction="Create a calendar event",
                user_input=event_text,
                expected_tool="CreateEvent",
                expected_parameters=params,
                difficulty="medium",
                category="event"
            ))

        return examples

    def generate_complex_examples(self, count: int = 10) -> List[ToolCallExample]:
        """Generate complex examples that might require reasoning"""
        examples = []

        # Ambiguous cases
        examples.append(ToolCallExample(
            instruction="Process the user request",
            user_input="John Smith (32) needs the sum of 15 and 27",
            expected_tool="CalculateMath",  # Should prioritize math despite extraction info
            expected_parameters={"operation": "add", "operands": [15, 27]},
            difficulty="hard",
            category="complex"
        ))

        # Context-dependent
        examples.append(ToolCallExample(
            instruction="Help the user",
            user_input="Translate 'Hello World' to Spanish",
            expected_tool="TranslateText",
            expected_parameters={"text": "Hello World", "target_language": "Spanish"},
            difficulty="medium",
            category="translation"
        ))

        # No tool needed
        examples.append(ToolCallExample(
            instruction="Respond to the user",
            user_input="What's the weather like?",
            expected_tool="",  # No tool needed
            expected_parameters={},
            difficulty="hard",
            category="no_tool"
        ))

        return examples[:count]

    def generate_dataset(self, size: int = 100) -> List[ToolCallExample]:
        """Generate a complete dataset with diverse examples"""
        dataset = []

        # Balance different types
        dataset.extend(self.generate_extraction_examples(size // 5))
        dataset.extend(self.generate_math_examples(size // 5))
        dataset.extend(self.generate_search_examples(size // 5))
        dataset.extend(self.generate_event_examples(size // 5))
        dataset.extend(self.generate_complex_examples(size // 5))

        # Shuffle for variety
        random.shuffle(dataset)

        return dataset

    def save_dataset(self, dataset: List[ToolCallExample], filepath: str):
        """Save dataset to JSON file"""
        data = [asdict(example) for example in dataset]

        with open(filepath, 'w') as f:
            json.dump({
                "tools": self.tools,
                "examples": data
            }, f, indent=2)

        print(f"Saved {len(dataset)} examples to {filepath}")

    def generate_llm_created_examples(self, count: int = 20) -> List[ToolCallExample]:
        """Use the larger model to generate more diverse examples"""
        if not self.llm:
            raise ValueError("Model not available for LLM generation")

        examples = []

        # More specific prompt to get better formatted responses
        prompt_template = """Given this tool definition:
{tool_def}

Create a realistic {difficulty} difficulty example where a user makes a request that requires this tool.

Provide the response in exactly this format:
User: [write the user's request here]
Parameters: {{"key": "value"}}

For example, if the tool is ExtractUserInfo and the user says "My name is John and I'm 25 years old":
User: My name is John and I'm 25 years old
Parameters: {{"name": "John", "age": 25}}

Now generate an example:
"""

        # Generate examples for each tool
        examples_per_tool = max(1, count // len(self.tools))

        for tool in self.tools:
            tool_name = tool["function"]["name"]
            tool_def = json.dumps(tool["function"], indent=2)

            # Generate examples of different difficulties
            for difficulty in ["easy", "medium"]:
                if len(examples) >= count:
                    break

                prompt = prompt_template.format(
                    tool_name=tool_name,
                    difficulty=difficulty,
                    tool_def=tool_def
                )

                try:
                    # Generate based on LLM type
                    if self.llm_type == "llama_cpp":
                        response = self.llm(
                            prompt,
                            max_tokens=256,
                            temperature=0.7,
                            stop=["\n\n", "Tool:", "Example:"]
                        )
                        response_text = response['choices'][0]['text']
                    else:
                        response_text = self.llm.generate(prompt)

                    # Parse the response more carefully
                    user_request = ""
                    params = {}

                    lines = response_text.strip().split('\n')
                    for i, line in enumerate(lines):
                        if 'User:' in line:
                            user_request = line.split('User:', 1)[1].strip()
                        elif 'Parameters:' in line:
                            # Get everything after "Parameters:"
                            params_str = line.split('Parameters:', 1)[1].strip()
                            # Also check if parameters continue on next lines
                            if i + 1 < len(lines) and not lines[i + 1].startswith('User:'):
                                # Might be multi-line JSON
                                remaining = '\n'.join(lines[i+1:])
                                if remaining.strip().startswith('{') or remaining.strip().startswith('}'):
                                    params_str += '\n' + remaining

                            try:
                                params = json.loads(params_str)
                            except json.JSONDecodeError:
                                # Try to extract JSON from the string
                                import re
                                json_match = re.search(r'\{[^}]*\}', params_str)
                                if json_match:
                                    try:
                                        params = json.loads(json_match.group())
                                    except:
                                        pass

                    # Validate and fix parameters based on tool definition
                    if user_request and tool_name != "":
                        # Ensure we have valid parameters
                        if not params:
                            # Generate meaningful parameters based on the user request
                            params = self._extract_params_from_request(
                                user_request,
                                tool["function"]["parameters"]
                            )

                        examples.append(ToolCallExample(
                            instruction="Process the user request",
                            user_input=user_request,
                            expected_tool=tool_name,
                            expected_parameters=params,
                            difficulty=difficulty,
                            category="llm_generated"
                        ))

                except Exception as e:
                    print(f"    Error generating example for {tool_name}: {e}")
                    continue

        return examples[:count]

    def _extract_params_from_request(self, request: str, param_schema: dict) -> dict:
        """Extract parameters from user request based on schema"""
        params = {}
        props = param_schema.get("properties", {})

        # Simple extraction based on tool type
        request_lower = request.lower()

        # For ExtractUserInfo
        if "name" in props:
            # Look for names (simple heuristic)
            import re
            name_patterns = [
                r"(?:name is|I'm|I am|called) ([A-Z][a-z]+)",
                r"([A-Z][a-z]+ [A-Z][a-z]+)",  # Full names
            ]
            for pattern in name_patterns:
                match = re.search(pattern, request)
                if match:
                    params["name"] = match.group(1)
                    break

        if "age" in props:
            # Look for ages
            age_match = re.search(r"(\d{1,3}) (?:years old|yo)", request)
            if age_match:
                params["age"] = int(age_match.group(1))

        if "email" in props:
            # Look for email
            email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", request)
            if email_match:
                params["email"] = email_match.group()

        # For CalculateMath
        if "operation" in props:
            if any(word in request_lower for word in ["add", "plus", "sum"]):
                params["operation"] = "add"
            elif any(word in request_lower for word in ["subtract", "minus"]):
                params["operation"] = "subtract"
            elif any(word in request_lower for word in ["multiply", "times"]):
                params["operation"] = "multiply"
            elif any(word in request_lower for word in ["divide"]):
                params["operation"] = "divide"

        if "operands" in props:
            # Extract numbers
            numbers = re.findall(r'\b\d+\b', request)
            if numbers:
                params["operands"] = [int(n) for n in numbers]

        # For SearchDatabase
        if "query" in props:
            params["query"] = request  # Use the whole request as query

        # For CreateEvent
        if "title" in props:
            if "meeting" in request_lower:
                params["title"] = "Meeting"
            elif "lunch" in request_lower:
                params["title"] = "Lunch"
            else:
                params["title"] = "Event"

        if "date" in props:
            params["date"] = "2024-01-15"  # Default date

        # For TranslateText
        if "text" in props:
            # Extract quoted text if present
            quote_match = re.search(r"['\"]([^'\"]+)['\"]", request)
            if quote_match:
                params["text"] = quote_match.group(1)

        if "target_language" in props:
            languages = ["Spanish", "French", "German", "Chinese", "Japanese"]
            for lang in languages:
                if lang.lower() in request_lower:
                    params["target_language"] = lang
                    break

        # Ensure required fields have at least default values
        required = param_schema.get("required", [])
        for field in required:
            if field not in params and field in props:
                prop_type = props[field].get("type", "string")
                if prop_type == "string":
                    params[field] = "example"
                elif prop_type == "integer":
                    params[field] = 1
                elif prop_type == "array":
                    params[field] = []
                elif prop_type == "object":
                    params[field] = {}

        return params
