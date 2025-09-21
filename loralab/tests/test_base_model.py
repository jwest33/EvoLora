"""Test base model performance on different math tasks"""

import torch
from loralab.core.unsloth_manager import UnslothModelManager
from loralab.datasets.dataset_loader import DatasetLoader

def test_model_on_datasets():
    """Test the base Gemma model on various math datasets"""

    # Initialize model
    print("Loading Gemma3-270M base model...")
    model_manager = UnslothModelManager(
        model_path="unsloth/gemma-3-270m-it",
        quantization="none",
        max_seq_length=512
    )
    model_manager.load_model()

    # Test different datasets
    datasets_to_test = [
        ('simple-math', 50),       # Simple arithmetic
        ('arithmetic-mul', 50),    # Multiplication only
        ('gsm8k', 50),            # Word problems
    ]

    loader = DatasetLoader()

    for dataset_name, num_samples in datasets_to_test:
        print(f"\n{'='*60}")
        print(f"Testing on {dataset_name} ({num_samples} samples)")
        print('='*60)

        try:
            # Load dataset
            data = loader.load_dataset(dataset_name, train_size=0, eval_size=num_samples)
            eval_data = data['eval']

            # Test model
            correct = 0
            for i, item in enumerate(eval_data[:10]):  # Show first 10
                question = item['question']
                true_answer = item['answer']

                # Generate answer
                prompt = f"Calculate: {question}\nAnswer:"
                response = model_manager.generate(
                    prompt,
                    max_new_tokens=50,
                    temperature=0.1,  # Low temp for deterministic
                )

                # Simple check - does response contain the answer?
                if str(true_answer).strip() in response:
                    correct += 1
                    result = "[OK]"
                else:
                    result = "[WRONG]"

                if i < 3:  # Show first 3 examples
                    print(f"\nExample {i+1}:")
                    print(f"  Q: {question[:100]}")
                    print(f"  True: {true_answer}")
                    print(f"  Pred: {response[-50:]}")  # Last 50 chars
                    print(f"  {result}")

            accuracy = (correct / min(10, len(eval_data))) * 100
            print(f"\nAccuracy on first 10: {accuracy:.1f}%")

        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")

if __name__ == "__main__":
    test_model_on_datasets()
