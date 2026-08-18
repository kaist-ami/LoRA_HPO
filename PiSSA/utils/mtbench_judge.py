"""
MT-Bench 스타일 Judge: conversation_response.jsonl 각 줄의 model output을 1~10 점수로 평가.
- OPENAI_API_KEY 환경변수 필요 (또는 OPENAI_BASE_URL로 호환 API 사용).
- 입력: type, query, output, answer 가 있는 jsonl
- 출력: 동일 jsonl에 "score" (1~10 float) 필드 추가 (in-place 또는 --output_file 지정 시 해당 파일에 저장).
"""
import argparse
import json
import os
import re
import sys
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("openai 패키지 필요: pip install openai", file=sys.stderr)
    sys.exit(1)

DEFAULT_JUDGE_PROMPT = """You are an expert judge. Rate the following assistant response on a scale of 1 to 10 for helpfulness, relevance, and quality. Consider clarity and whether the response addresses the user's question.

User question:
{query}

Assistant response:
{output}

Reply with ONLY a single number between 1 and 10 (e.g. 7). No explanation."""

def extract_score(text: str) -> Optional[float]:
    """응답 텍스트에서 1~10 점수 추출."""
    text = text.strip()
    matches = re.findall(r"\b(10|\d)(?:\.\d+)?\s*$", text)
    if matches:
        return max(1.0, min(10.0, float(matches[-1])))
    matches = re.findall(r"\b(10|\d)(?:\.\d+)?\b", text)
    if matches:
        return max(1.0, min(10.0, float(matches[-1])))
    return None

def run_judge_on_line(client: OpenAI, line_data: dict, model: str, prompt_template: str) -> float:
    """한 줄(한 샘플)에 대해 judge API 호출 후 1~10 점수 반환."""
    query = line_data.get("query", "") or ""
    output = line_data.get("output", "") or ""
    prompt = prompt_template.format(query=query, output=output)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        content = (resp.choices[0].message.content or "").strip()
        score = extract_score(content)
        if score is not None:
            return score
    except Exception as e:
        print(f"[mtbench_judge] API error: {e}", file=sys.stderr)
    return 5.0

def main():
    parser = argparse.ArgumentParser(description="MT-Bench style judge: add 'score' (1-10) to conversation jsonl")
    parser.add_argument("--input_file", "-i", required=True, help="conversation_response.jsonl path")
    parser.add_argument("--output_file", "-o", default=None, help="출력 jsonl (미지정 시 입력 파일을 덮어씀)")
    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("MTBENCH_JUDGE_MODEL", "gpt-4"),
        help="Judge model (e.g. gpt-4o, gpt-4o-mini). env: MTBENCH_JUDGE_MODEL",
    )
    parser.add_argument("--max_lines", type=int, default=None, help="처리할 최대 줄 수 (디버그용)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        print("OPENAI_API_KEY 환경변수를 설정해 주세요.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url if base_url else None)
    out_path = args.output_file or args.input_file
    prompt_template = DEFAULT_JUDGE_PROMPT

    lines = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
            if args.max_lines and len(lines) >= args.max_lines:
                break

    if not lines:
        print("No lines to judge.", file=sys.stderr)
        sys.exit(0)

    for i, data in enumerate(lines):
        if "score" in data:
            continue
        score = run_judge_on_line(client, data, args.model, prompt_template)
        data["score"] = round(score, 2)
        if (i + 1) % 10 == 0:
            print(f"Judged {i + 1}/{len(lines)} ...", file=sys.stderr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for data in lines:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"Wrote {len(lines)} lines with scores to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
