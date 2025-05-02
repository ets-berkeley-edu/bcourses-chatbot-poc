import os
import sys
import click

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_chatbot import test_qa_accuracy

@click.command()
@click.option('--dataset', default='chabot-test-questions', help='Name of the LangSmith dataset to use')
def run_evaluation(dataset):
    """Run chatbot evaluation using specified dataset"""
    click.echo(f"Running evaluation using dataset: {dataset}")
    test_qa_accuracy()
    click.echo("Evaluation complete")

if __name__ == '__main__':
    run_evaluation()