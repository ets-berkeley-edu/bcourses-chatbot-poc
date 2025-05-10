import boto3
import logging
from langsmith import Client
from typing import Optional, Dict, Any, List, Tuple
from langchain.evaluation import EvaluatorType
from langchain.evaluation.schema import PairwiseStringEvaluator
from app.main import initialize_llm, initialize_retriever, create_prompt_templates
from app.config_manager import ConfigManager
from langchain.chains import ConversationalRetrievalChain

logger = logging.getLogger(__name__)

def setup_qa_chain():
    """Setup the QA chain for testing"""
    ConfigManager.initialize()
    config = ConfigManager.config

    session = boto3.Session()
    retriever = initialize_retriever(session, config)
    llm = initialize_llm(session, config)
    few_shot_prompt, condense_question_prompt = create_prompt_templates()

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": few_shot_prompt},
        condense_question_prompt=condense_question_prompt,
    )

class RTLServiceEvaluator(PairwiseStringEvaluator):
    """Custom evaluator for RTL Service responses"""

    def __init__(self):
        super().__init__()

    def _evaluate_string_pairs(
        self,
        prediction: str,
        reference: str,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate the prediction against the reference answer.
        Returns a score and reasoning.
        """
        # Check for required metadata
        has_kb_url = "kb_url" in prediction.lower()
        has_kb_number = "kb_number" in prediction.lower()
        metadata_score = 1.0 if (has_kb_url and has_kb_number) else 0.0

        # Check content similarity (basic check - can be enhanced)
        reference_points = [point.strip() for point in reference.lower().split('.') if point.strip()]
        key_points_present = all(point in prediction.lower() for point in reference_points)
        content_score = 1.0 if key_points_present and reference_points else (1.0 if not reference_points else 0.0)

        # Calculate final score (equal weighting)
        final_score = (metadata_score + content_score) / 2.0

        reasoning = (
            f"Metadata present: {metadata_score:.1f} "
            f"({'kb_url' if has_kb_url else ''} "
            f"{'kb_number' if has_kb_number else ''}).\n"
            f"Content accuracy: {content_score:.1f}"
        )

        return {
            "score": final_score,
            "reasoning": reasoning,
            "metadata_score": metadata_score,
            "content_score": content_score
        }

def test_qa_accuracy():
    """Test the QA chain's accuracy using LangSmith"""
    client = Client()
    qa_chain = setup_qa_chain()

    def qa_chain_wrapper(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        chat_history: List[Tuple[str, str]] = []
        if "chat_history" in inputs:
            raw_history: List[Dict[str, str]] = inputs["chat_history"]
            for msg in raw_history:
                if msg["type"] == "human":
                    chat_history.append((msg["content"], ""))
                elif msg["type"] == "ai" and chat_history:
                    chat_history[-1] = (chat_history[-1][0], msg["content"])
                elif msg["type"] == "ai" and not chat_history:
                    chat_history.append(("", msg["content"]))

        result = qa_chain.invoke({"question": question, "chat_history": chat_history})
        return {"response": result["answer"]}

    # Create custom evaluator instance
    custom_evaluator = RTLServiceEvaluator()

    import time
    timestamp = int(time.time())
    project_name = f"chatbot-evaluation-{timestamp}"

    experiment_results = client.run_on_dataset(
        dataset_name="chabot-test-questions",
        llm_or_chain_factory=qa_chain_wrapper,
        evaluators=[custom_evaluator],
        project_name=project_name
    )

    logger.info(f"Experiment completed. Results: {experiment_results}")
    return experiment_results