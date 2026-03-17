import os
import json
import argparse
import ast
import time
from typing import Any, List, Optional, Set, cast

from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract answer tokens from consistent responses."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to samples files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/answer_tokens.jsonl",
        help="Path to save processed results",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="data/activations",
        help="Path to the target model tokenizer",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["llm", "synthetic-output"],
        default="llm",
        help="Use GPT extraction or build zero-cost synthetic records for output-token training.",
    )

    # LLM Extractor Config
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API Key (required for --strategy llm)",
    )
    parser.add_argument(
        "--base_url", type=str, default="https://api.openai.com/v1", help="API Base URL"
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-4o", help="LLM for extraction"
    )

    return parser.parse_args()


# ==========================================
# LLM Prompt Templates
# ==========================================

USER_INPUT_TEMPLATE = """Question: {question}
Response: {response}
Tokenized Response: {response_tokens}
Please help extract the "answer tokens" from all tokens, removing all redundant information, and the tokens you return must be part of the input Tokenized Response list."""

EXAMPLE_MESSAGES = [
    {
        "role": "user",
        "content": "Question: What is the correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator.\nResponse: The correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator is the Spirit of Ecstasy.\nTokenized Response: ['▁The', '▁correct', '▁name', '▁for', '▁the', '▁\"', 'F', 'lying', '▁Lady', '\"', '▁or', 'nament', '▁on', '▁a', '▁Roll', 's', '▁Roy', 'ce', '▁radi', 'ator', '▁is', '▁the', '▁Spirit', '▁of', '▁Ec', 'st', 'asy', '.']\nPlease help extract the \"answer tokens\" from all tokens, removing all redundant information, and the tokens you return must form a continuous segment of the input Tokenized Response list.",
    },
    {"role": "assistant", "content": "['▁the', '▁Spirit', '▁of', '▁Ec', 'st', 'asy']"},
]

# ==========================================
# Extraction Logic
# ==========================================


class AnswerTokenExtractor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.client = None

        if args.strategy == "llm":
            if not args.api_key:
                raise ValueError("--api_key is required when --strategy llm.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path, trust_remote_code=True
            )
            self.client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    def get_tokenized_list(self, text: str) -> List[str]:
        assert self.tokenizer is not None
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return tokens

    def parse_token_list(self, reply: str) -> Optional[List[str]]:
        """Accept JSON, Python-style lists, and fenced wrapper text."""
        reply = reply.strip()
        candidates = [reply]

        if reply.startswith("```"):
            stripped = [
                line
                for line in reply.splitlines()
                if not line.strip().startswith("```")
            ]
            if stripped:
                candidates.append("\n".join(stripped).strip())

        start = reply.find("[")
        end = reply.rfind("]")
        if start != -1 and end != -1 and end >= start:
            candidates.append(reply[start : end + 1].strip())

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(candidate)
                except Exception:
                    continue
                if isinstance(parsed, list) and all(
                    isinstance(token, str) for token in parsed
                ):
                    return parsed
        return None

    def coerce_to_contiguous_span(
        self, tokens: List[str], extracted: List[str]
    ) -> Optional[List[str]]:
        """Project a valid ordered subsequence onto the minimal contiguous span."""
        if not extracted:
            return None

        span_len = len(extracted)
        for start in range(len(tokens) - span_len + 1):
            if tokens[start : start + span_len] == extracted:
                return extracted

        positions = []
        search_start = 0
        for token in extracted:
            try:
                position = tokens.index(token, search_start)
            except ValueError:
                return None
            positions.append(position)
            search_start = position + 1

        return tokens[positions[0] : positions[-1] + 1]

    def extract_via_llm(
        self, question: str, response: str, tokens: List[str]
    ) -> Optional[List[str]]:
        """Request LLM to select tokens from the tokenized list."""
        prompt = USER_INPUT_TEMPLATE.format(
            question=question, response=response, response_tokens=str(tokens)
        )

        for attempt in range(3):
            try:
                assert self.client is not None
                messages = cast(
                    Any,
                    EXAMPLE_MESSAGES + [{"role": "user", "content": prompt}],
                )
                completion = self.client.chat.completions.create(
                    model=self.args.llm_model,
                    messages=messages,
                    temperature=0.0,
                )
                reply_content = completion.choices[0].message.content
                if reply_content is None:
                    raise ValueError("Empty extractor response.")
                reply = reply_content.strip()
                extracted = self.parse_token_list(reply)
                if not extracted:
                    raise ValueError(
                        f"Could not parse token list from reply: {reply!r}"
                    )

                contiguous_span = self.coerce_to_contiguous_span(tokens, extracted)
                if contiguous_span:
                    return contiguous_span
            except Exception as e:
                print(f"Extraction failed (attempt {attempt + 1}): {e}")
                time.sleep(1)
        return None

    def load_processed_ids(self) -> Set[str]:
        """Resume from existing output file."""
        if not os.path.exists(self.args.output_path):
            return set()
        ids = set()
        with open(self.args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ids.update(json.loads(line).keys())
                except json.JSONDecodeError:
                    continue
        return ids

    def run(self):
        processed_ids = self.load_processed_ids()

        with (
            open(self.args.input_path, "r", encoding="utf-8") as f_in,
            open(self.args.output_path, "a", encoding="utf-8") as f_out,
        ):
            for line in tqdm(f_in, desc="Processing tokens"):
                data = json.loads(line)
                qid = list(data.keys())[0]
                content = data[qid]

                if qid in processed_ids:
                    continue

                # Ensure all 10 responses have the same judge outcome (true/false)
                judges = content["judges"]
                if len(set(judges)) != 1 or "uncertain" in judges or "error" in judges:
                    continue

                # Take the most frequent response as representative
                responses = content["responses"]
                rep_response = max(set(responses), key=responses.count)

                if self.args.strategy == "synthetic-output":
                    tokenized_list = []
                    answer_tokens = []
                else:
                    tokenized_list = self.get_tokenized_list(rep_response)
                    answer_tokens = self.extract_via_llm(
                        content["question"], rep_response, tokenized_list
                    )

                if self.args.strategy == "synthetic-output" or answer_tokens:
                    result = {
                        qid: {
                            "question": content["question"],
                            "response": rep_response,
                            "tokenized_response": tokenized_list,
                            "answer_tokens": answer_tokens,
                            "judge": judges[
                                0
                            ],  # Consistently correct or consistently hallucinated
                        }
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()


if __name__ == "__main__":
    args = parse_args()
    extractor = AnswerTokenExtractor(args)
    extractor.run()
