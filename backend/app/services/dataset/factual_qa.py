import asyncio
import json
import logging
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QAPair(BaseModel):
    """A single question–answer pair used for factual QA datasets."""
    question: str = Field(..., description="User question to reveal the fact")
    answer: str = Field(..., description="Character's first-person answer confirming the fact")


async def extract_and_simplify_facts(client, character: Dict[str, Any], max_facts: int = 20) -> List[str]:
    """
    Uses an LLM to distill a character card into a list of simple, verifiable facts.
    This is the foundation of the factual Q&A generation process.
    """
    logger.info(f"Extracting and simplifying facts for character: {character.get('name', 'Unknown')}")
    
    # Consolidate key character information
    char_name = character.get("name", "the character")
    description = character.get("description", "")
    personality = character.get("personality", "")
    scenario = character.get("scenario", "")
    example_dialogue = character.get("mes_example", "")

    source_text = f"""
    Character Name: {char_name}
    Description: {description}
    Personality: {personality}
    Scenario: {scenario}
    Example Dialogue: {example_dialogue}
    """

    base_prompt_template = (
        "You are given background information about a fictional role-play character named '{char_name}'. "
        "Your task is to list clear, easily verifiable FACTS about this character only — ignore any unrelated "
        "meaning of the name (e.g. the sport 'cricket').\n\n"
        "Rules:\n"
        "1. Each fact must be a single, declarative sentence with no introductory phrases.\n"
        "2. Focus on concrete traits, skills, relationships, history, motivations, appearance, etc.\n"
        "3. Avoid fluff, figurative language, or speculation.\n"
        "4. Produce as many distinct facts as possible (up to {max_facts}).\n"
        "5. Output ONLY a numbered list (e.g. '1. ...'). DO NOT add any other text.\n\n"
        "Character information:\n---\n{card}\n---\n\n"
        "List the facts now:"
    )

    prompt = base_prompt_template.format(char_name=char_name, max_facts=max_facts, card=textwrap.dedent(source_text))

    try:
        response = await client.generate(prompt, temperature=0.2, max_tokens=1024)
        
        # Parse the numbered list
        facts = re.findall(r'^\s*\d+\.\s*(.*)', response, re.MULTILINE)
        
        if not facts:
            logger.warning("LLM failed to return a numbered list of facts. Falling back to splitting by newline.")
            facts = [line.strip() for line in response.split('\n') if line.strip()]

        logger.info(f"Extracted {len(facts)} facts for {char_name}.")
        return facts[:max_facts]

    except Exception as e:
        logger.error(f"Failed to extract facts from character card: {e}", exc_info=True)
        return []

    # --- Simple retry if extraction is unexpectedly small ---
    if len(facts) <= 1:
        logger.info("⚠️  Very few facts extracted, retrying with reinforced instructions…")
        retry_prompt = (
            "IMPORTANT REMINDER: The name '{char_name}' refers to the CHARACTER described above. "
            "Extract facts **about that character only** (ignore anything about sports)."\
        ).format(char_name=char_name) + "\n\n" + prompt

        retry_response = await client.generate(retry_prompt, temperature=0.2, max_tokens=1024)
        retry_facts = re.findall(r'^\s*\d+\.\s*(.*)', retry_response, re.MULTILINE)
        if len(retry_facts) > len(facts):
            facts = retry_facts

        logger.info(f"Extracted {len(facts)} facts for {char_name}.")
        return facts[:max_facts]


async def generate_factual_qa_variations(
    client,
    fact: str,
    character: Dict[str, Any],
    *,
    num_variations: int = 3,
    length_category: str = "short",
) -> List[Dict[str, str]]:
    """Generate Q&A pairs for a fact with a targeted length style.

    length_category: 'short'  (≈ < 20 words answer)
                     'medium' (≈ 20-60 words answer)
                     'long'   (≈ 60-120 words answer)
    """

    char_name = character.get("name", "the character")

    # Word-count guidance
    if length_category == "short":
        q_words = "under 12 words"
        a_words = "under 20 words"
    elif length_category == "medium":
        q_words = "around 15-25 words"
        a_words = "around 30-60 words"
    else:  # long
        q_words = "30-40 words"
        a_words = "80-120 words"

    # JSON schema for an array of QAPair items
    qa_schema = {
        "type": "array",
        "items": QAPair.model_json_schema(),
        "minItems": num_variations,
        "maxItems": num_variations,
    }

    prompt = (
        f"You are role-playing as the user who wants to learn about {char_name}.\n"
        f"Using the FACT below, craft {num_variations} distinct question & answer pairs.\n"
        f"• Question length: {q_words}.\n"
        f"• Answer length: {a_words}, written in first-person as {char_name}.\n"
        "• Make the wording natural and engaging.\n"
        "• Do not reveal meta reasoning.\n\n"
        f"FACT: {fact}\n\n"
        "Return ONLY the JSON array that matches the provided schema. Avoid using markdown formatting for the json response."
    )

    try:
        response = await client.generate(
            prompt=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7,
            top_p=0.9,
            guided_json=qa_schema,
        )

        qa_pairs: List[Dict[str, str]] = json.loads(response)
        # Basic validation: ensure required keys are present
        validated_pairs: List[Dict[str, str]] = []
        for pair in qa_pairs:
            try:
                validated = QAPair.model_validate(pair).model_dump()
                validated_pairs.append(validated)
            except Exception as val_err:
                logger.debug(f"Validation failed for pair {pair}: {val_err}")
        return validated_pairs
    except Exception as e:
        logger.error(
            f"Failed to generate structured Q&A variations for fact '{fact}': {e}",
            exc_info=True,
        )
        return []


async def generate_factual_qa_dataset(
    client,
    character: Dict[str, Any], 
    num_facts_to_use: int = 15, 
    variations_per_fact: int = 3,
    progress_callback: Optional[Callable] = None,
    stage_callback: Optional[Callable] = None,
    save_dataset_func: Optional[Callable] = None
) -> List[Dict[str, Any]]:
    """
    Generates a high-quality dataset consisting of varied Q&A pairs based on
    distilled facts from the character card. This method prioritizes factual accuracy
    and reinforcement over narrative generation.
    """
    char_name = character.get("name", "Unknown")
    logger.info(f"Starting factual Q&A dataset generation for {char_name}.")
    
    if stage_callback:
        stage_callback({"message": "Step 1/3: Extracting and simplifying facts..."})

    facts = await extract_and_simplify_facts(client, character, max_facts=num_facts_to_use)
    if not facts:
        logger.error("No facts could be extracted. Aborting dataset generation.")
        if stage_callback:
            stage_callback({"message": "Error: Could not extract facts."})
        return []

    if stage_callback:
        stage_callback({"message": f"Step 2/3: Generating Q&A variations for {len(facts)} facts..."})

    dataset: List[Dict[str, Any]] = []
    tasks = []

    # length distribution – tweakable
    length_plan = [
        ("short", 2),  # two short pairs per fact
        ("medium", 1),
        ("long", 1),
    ]

    for fact in facts:
        for length_cat, count in length_plan:
            task = generate_factual_qa_variations(
                client,
                fact,
                character,
                num_variations=count,
                length_category=length_cat,
            )
            tasks.append(task)
    
    total_tasks = len(tasks)
    completed_tasks = 0
    
    for future in asyncio.as_completed(tasks):
        try:
            qa_pairs = await future
            for pair in qa_pairs:
                # Create the standard message format
                sample = {
                    "messages": [
                        {"role": "system", "content": ""}, # Factual datasets often have no system prompt
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]},
                    ]
                }
                dataset.append(sample)
            
            completed_tasks += 1
            if progress_callback:
                progress_callback(completed_tasks / total_tasks)

        except Exception as e:
            logger.error(f"Error processing Q&A generation task: {e}", exc_info=True)

    logger.info(f"Generated {len(dataset)} factual Q&A pairs for {char_name}.")
    if stage_callback:
        stage_callback({"message": f"Step 3/3: Finalizing dataset..."})

    # Save the dataset if function provided
    if save_dataset_func:
        save_dataset_func(character, dataset, metadata={"generation_method": "factual_qa"})

    return dataset