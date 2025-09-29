from __future__ import annotations

import json, re
from typing import cast

from loguru import logger
from pydantic_ai import Agent, Tool
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings
from pydantic_ai.models.openai import (
    OpenAIModel,
    OpenAIResponsesModel,
)
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.google_vertex import GoogleVertexProvider, VertexAiRegion
from pydantic_ai.providers.openai import OpenAIProvider

from ..config import detect_provider_from_model, settings
from ..prompts import (
    CYPHER_SYSTEM_PROMPT,
    LOCAL_CYPHER_SYSTEM_PROMPT,
    RAG_ORCHESTRATOR_SYSTEM_PROMPT,
)

 

# Optional supervisor agent that critiques plans and enforces graph-first behavior
REASONING_SUPERVISOR_SYSTEM_PROMPT = """
You are a senior reasoning supervisor for a codebase RAG agent.

Your responsibilities:
- Critique and improve the agent's plan before execution.
- Enforce graph-first retrieval and safe Cypher rules.
- Detect when results are too broad and propose narrowed follow-ups (path prefix, extension, name keyword, decorator, or relationship filter).
- If results exist, force evidence-based synthesis with citations.

Decision policy:
1) If the next step is not graph-first when it should be, rewrite it.
2) If Cypher candidate lacks label-specific MATCH, narrowing, or LIMIT ≤ 50, reject and request regeneration with a hint.
3) If results are 0, propose a narrowed query.
4) If results are > 500 in estimate, require adding constraints.
5) If answering a "How does X work?", require at least one file read and snippet before synthesis.

Return a compact JSON with fields: { "critique": "...", "revised_plan": ["..."], "next_actions": [{"tool":"...","args":{...}}] }.
"""

def create_reasoning_supervisor() -> Agent:
    orchestrator_model_id = settings.active_orchestrator_model
    provider_name = detect_provider_from_model(orchestrator_model_id)
    model_settings = None
    if provider_name == "gemini":
        if settings.GEMINI_PROVIDER == "vertex":
            provider = GoogleVertexProvider(
                project_id=settings.GCP_PROJECT_ID,
                region=cast(VertexAiRegion, settings.GCP_REGION),
                service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
            )
        else:
            provider = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)  # type: ignore
        if settings.GEMINI_THINKING_BUDGET is not None:
            model_settings = GeminiModelSettings(
                gemini_thinking_config={"thinking_budget": int(settings.GEMINI_THINKING_BUDGET)}
            )
        gen_cfg = settings.get_orchestrator_generation_config()
        llm = GeminiModel(
            orchestrator_model_id,
            provider=provider,
            temperature=gen_cfg.get("temperature", 0.2),
            top_p=gen_cfg.get("top_p", 1.0),
            max_output_tokens=int(gen_cfg.get("max_tokens", 2000)),
        )
    elif provider_name == "local":
        llm = OpenAIModel(  # type: ignore
            orchestrator_model_id,
            provider=OpenAIProvider(
                api_key=settings.LOCAL_MODEL_API_KEY,
                base_url=str(settings.LOCAL_MODEL_ENDPOINT),
            ),
        )
    else:
        llm = OpenAIResponsesModel(
            orchestrator_model_id,
            provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY),
        )
    return Agent(
        model=llm,
        system_prompt=REASONING_SUPERVISOR_SYSTEM_PROMPT,
        output_type=dict,
        model_settings=model_settings,
        output_retries=settings.LLM_OUTPUT_RETRIES,
    )


class LLMGenerationError(Exception):
    """Custom exception for LLM generation failures."""

    pass


def _clean_cypher_response(response_text: str) -> str:
    """Utility to clean up common LLM formatting artifacts from a Cypher query."""
    text = response_text.strip()
    # Strip code fences
    if text.startswith("```"):
        text = text.strip("`\n ")
        # Remove optional language label like json or cypher
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if first.lower() in {"json", "cypher"}:
                text = rest
    # Remove backticks and leading language tokens
    query = text.replace("`", "").lstrip()
    if query.lower().startswith("json"):
        query = query[4:].lstrip()
    if query.lower().startswith("cypher"):
        query = query[6:].lstrip()
    if not query.endswith(";"):
        query += ";"
    return query


class CypherGenerator:
    """Generates Cypher queries from natural language."""

    def __init__(self) -> None:
        try:
            model_settings = None

            # Get active cypher model and detect its provider
            cypher_model_id = settings.active_cypher_model
            cypher_provider = detect_provider_from_model(cypher_model_id)

            # Configure model based on detected provider
            if cypher_provider == "gemini":
                if settings.GEMINI_PROVIDER == "vertex":
                    provider = GoogleVertexProvider(
                        project_id=settings.GCP_PROJECT_ID,
                        region=cast(VertexAiRegion, settings.GCP_REGION),
                        service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
                    )
                else:
                    provider = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)  # type: ignore

                if settings.GEMINI_THINKING_BUDGET is not None:
                    model_settings = GeminiModelSettings(
                        gemini_thinking_config={
                            "thinking_budget": int(settings.GEMINI_THINKING_BUDGET)
                        }
                    )

                # Apply generation params
                gen_cfg = settings.get_cypher_generation_config()
                llm = GeminiModel(
                    cypher_model_id,
                    provider=provider,
                    temperature=gen_cfg.get("temperature", 0.0),
                    top_p=gen_cfg.get("top_p", 1.0),
                    max_output_tokens=int(gen_cfg.get("max_tokens", 800)),
                )
                system_prompt = CYPHER_SYSTEM_PROMPT
            elif cypher_provider == "openai":
                # OpenAIResponsesModel does not accept temperature/top_p/max_tokens in ctor
                llm = OpenAIResponsesModel(
                    cypher_model_id,
                    provider=OpenAIProvider(
                        api_key=settings.OPENAI_API_KEY,
                    ),
                )
                system_prompt = CYPHER_SYSTEM_PROMPT
            else:  # local
                # Local OpenAI-compatible models (Ollama) do not accept these gen args in ctor
                llm = OpenAIModel(  # type: ignore
                    cypher_model_id,
                    provider=OpenAIProvider(
                        api_key=settings.LOCAL_MODEL_API_KEY,
                        base_url=str(settings.LOCAL_MODEL_ENDPOINT),
                    ),
                )
                system_prompt = LOCAL_CYPHER_SYSTEM_PROMPT
            self.agent = Agent(
                model=llm,
                system_prompt=system_prompt,
                output_type=str,
                model_settings=model_settings,
                output_retries=settings.LLM_OUTPUT_RETRIES,
            )
        except Exception as e:
            raise LLMGenerationError(
                f"Failed to initialize CypherGenerator: {e}"
            ) from e
            
    @staticmethod
    def _score_candidate(cypher: str) -> int:
        """
        Heuristic scoring to prefer scale-safe, precise queries.
        +2 LIMIT present (<= 100)
        +2 label-specific MATCH (Class|Function|Method|File|Module)
        +2 has WHERE with narrowing (STARTS WITH / extension / name/qualified_name / decorators / relationship pattern)
        +1 returns aliased properties name/path/qualified_name
        +1 contains toLower(...)
        -2 uses MATCH (n) with no WHERE
        """
        score = 0
        up = cypher.upper()
        if " LIMIT " in up: score += 2
        if re.search(r"MATCH\s*\(\s*\w+\s*:\s*(Class|Function|Method|File|Module)", up): score += 2
        narrow = any(k in up for k in [" STARTS WITH ", " EXTENSION ", "QUALIFIED_NAME", "DECORATORS", " CONTAINS "])
        if narrow: score += 2
        if all(k in up for k in [" RETURN ", " AS "]): score += 1
        if "TOLOWER" in up: score += 1
        if re.search(r"MATCH\s*\(\s*\w+\s*\)", up) and " WHERE " not in up: score -= 2
        return score

    @staticmethod
    def _pick_best_candidate(payload: str) -> str | None:
        try:
            text = payload.strip()
            # Strip code fences
            if text.startswith("```"):
                text = text.strip("`\n ")
                if "\n" in text:
                    first, rest = text.split("\n", 1)
                    if first.lower() in {"json", "cypher"}:
                        text = rest
            # Remove leading language tokens
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
            data = json.loads(text)
            cands = data.get("candidates", [])
            if not isinstance(cands, list) or not cands:
                return None
            best = max(cands, key=lambda c: CypherGenerator._score_candidate(c.get("cypher","")))
            return best.get("cypher")
        except Exception:
            return None

    async def generate_candidates(self, natural_language_query: str) -> list[str]:
        """
        Generate up to 3 Cypher candidates, best first. Each candidate is
        validated to include a label-specific MATCH, at least one narrowing
        predicate, and LIMIT <= 50. Falls back to a single cleaned query.
        """
        logger.info(f"[CypherGenerator] Generating candidates for: '{natural_language_query}'")
        try:
            result = await self.agent.run(natural_language_query)
            text = result.output if isinstance(result.output, str) else str(result.output)

            # Attempt multi-candidate JSON
            try:
                payload = text.strip()
                if payload.startswith("```"):
                    payload = payload.strip("`\n ")
                    if "\n" in payload:
                        first, rest = payload.split("\n", 1)
                        if first.lower() in {"json", "cypher"}:
                            payload = rest
                if payload.lower().startswith("json"):
                    payload = payload[4:].lstrip()
                data = json.loads(payload)
                cands = data.get("candidates", []) if isinstance(data, dict) else []
                cyphers: list[str] = []
                for c in cands:
                    cy = c.get("cypher") if isinstance(c, dict) else None
                    if isinstance(cy, str):
                        cyphers.append(_clean_cypher_response(cy))
                cyphers = sorted(cyphers, key=self._score_candidate, reverse=True)[:3]
                # Clamp LIMIT and validate
                validated: list[str] = []
                for q in cyphers:
                    q = re.sub(r"LIMIT\s+(\d+)", lambda m: f"LIMIT {min(50, int(m.group(1)))}", q, flags=re.IGNORECASE)
                    up = q.upper()
                    has_label_specific = re.search(r"MATCH\s*\(\s*\w+\s*:\s*(CLASS|FUNCTION|METHOD|FILE|MODULE)\b", up) is not None
                    has_narrow = any(k in up for k in [" STARTS WITH ", " EXTENSION ", "QUALIFIED_NAME", " DECORATORS ", " CONTAINS "]) or re.search(r"-\s*\[:\w+\]\s*->", up) is not None
                    if has_label_specific and has_narrow and " LIMIT " in up and " MATCH " in up:
                        validated.append(_clean_cypher_response(q))
                if validated:
                    logger.info(f"[CypherGenerator] Selected {len(validated)} Cypher candidate(s)")
                    return validated
            except Exception:
                pass

            # Fallback: single-query path
            cypher = _clean_cypher_response(text)
            up = cypher.upper()
            if "MATCH" not in up or "LIMIT" not in up:
                raise LLMGenerationError(f"LLM did not generate a valid query. Output: {text}")
            cypher = re.sub(r"LIMIT\s+(\d+)", lambda m: f"LIMIT {min(50, int(m.group(1)))}", cypher, flags=re.IGNORECASE)
            return [cypher]
        except Exception as e:
            logger.error(f"[CypherGenerator] Error: {e}")
            raise LLMGenerationError(f"Cypher generation failed: {e}") from e

    async def generate_candidates(self, natural_language_query: str) -> list[str]:
        """
        Returns up to 3 Cypher candidates (strings), best-scored first.
        Falls back to a single cleaned query if JSON candidates are not returned.
        """
        logger.info(f"[CypherGenerator] Generating candidates for: '{natural_language_query}'")
        try:
            result = await self.agent.run(natural_language_query)
            text = result.output if isinstance(result.output, str) else str(result.output)
            # Try parse as JSON multi-candidate payload
            try:
                payload = text.strip()
                if payload.startswith("```"):
                    payload = payload.strip("`\n ")
                    if "\n" in payload:
                        first, rest = payload.split("\n", 1)
                        if first.lower() in {"json", "cypher"}:
                            payload = rest
                if payload.lower().startswith("json"):
                    payload = payload[4:].lstrip()
                data = json.loads(payload)
                cands = data.get("candidates", []) if isinstance(data, dict) else []
                cyphers = []
                for c in cands:
                    cy = c.get("cypher") if isinstance(c, dict) else None
                    if isinstance(cy, str) and "MATCH" in cy.upper():
                        cyphers.append(_clean_cypher_response(cy))
                # Sort by heuristic score
                cyphers = sorted(cyphers, key=self._score_candidate, reverse=True)[:3]
                # Clamp LIMITs
                cyphers = [re.sub(r"LIMIT\s+(\d+)", lambda m: f"LIMIT {min(50, int(m.group(1)))}", q, flags=re.IGNORECASE) for q in cyphers]
                # Filter invalid
                filtered: list[str] = []
                for q in cyphers:
                    up = q.upper()
                    has_label_specific = re.search(r"MATCH\s*\(\s*\w+\s*:\s*(CLASS|FUNCTION|METHOD|FILE|MODULE)\b", up) is not None
                    has_narrow = any(k in up for k in [" STARTS WITH ", " EXTENSION ", "QUALIFIED_NAME", " DECORATORS ", " CONTAINS "]) or re.search(r"-\s*\[:\w+\]\s*->", up) is not None
                    if has_label_specific and has_narrow and " LIMIT " in up:
                        filtered.append(q)
                if filtered:
                    logger.info(f"[CypherGenerator] Using {len(filtered)} candidate query(ies)")
                    return filtered
            except Exception:
                pass

            # Fallback: single cleaned query
            single = _clean_cypher_response(text)
            up = single.upper()
            if "MATCH" not in up or "LIMIT" not in up:
                raise LLMGenerationError(f"LLM did not generate a valid query. Output: {text}")
            single = re.sub(r"LIMIT\s+(\d+)", lambda m: f"LIMIT {min(50, int(m.group(1)))}", single, flags=re.IGNORECASE)
            logger.info("[CypherGenerator] Falling back to single candidate")
            return [single]
        except Exception as e:
            logger.error(f"[CypherGenerator] Error generating candidates: {e}")
            raise LLMGenerationError(f"Cypher generation failed: {e}") from e

    async def generate(self, natural_language_query: str) -> str:
        logger.info(f"[CypherGenerator] Generating query for: '{natural_language_query}'")
        try:
            result = await self.agent.run(natural_language_query)
            text = result.output if isinstance(result.output, str) else str(result.output)

            # Try multi-candidate JSON first
            cypher = self._pick_best_candidate(text)
            if not cypher:
                # Fall back to single-query text path
                cypher = _clean_cypher_response(text)

            # Final sanity and gating
            up = cypher.upper()
            # Guard: if still looks like JSON, reject
            if up.strip().startswith("{") and 'CANDIDATES' in up:
                raise LLMGenerationError("Model returned JSON instead of a Cypher statement after parsing attempts")
            if "MATCH" not in up or "LIMIT" not in up:
                raise LLMGenerationError(f"LLM did not generate a valid query. Output: {text}")

            # Clamp LIMIT to <= 50
            try:
                cypher = _clean_cypher_response(cypher)
                cypher = re.sub(r"LIMIT\s+(\d+)", lambda m: f"LIMIT {min(50, int(m.group(1)))}", cypher, flags=re.IGNORECASE)
            except Exception:
                pass

            # Require label-specific MATCH and at least one narrowing predicate
            up2 = cypher.upper()
            has_label_specific = re.search(r"MATCH\s*\(\s*\w+\s*:\s*(CLASS|FUNCTION|METHOD|FILE|MODULE)\b", up2) is not None
            has_narrow = any(k in up2 for k in [" STARTS WITH ", " EXTENSION ", "QUALIFIED_NAME", " DECORATORS ", " CONTAINS "]) or re.search(r"-\s*\[:\w+\]\s*->", up2) is not None
            if not (has_label_specific and has_narrow):
                logger.info("[CypherGenerator] Regenerating due to missing label-specific MATCH or narrowing filter")
                hint = "Add label-specific MATCH and at least one narrowing filter (path STARTS WITH, extension, toLower(name|qualified_name) CONTAINS, decorator ANY(...), or a relationship hop). Keep LIMIT <= 50."
                try:
                    regen = await self.agent.run(f"{natural_language_query}\n\nHint: {hint}")
                    text2 = regen.output if isinstance(regen.output, str) else str(regen.output)
                    cypher2 = self._pick_best_candidate(text2) or _clean_cypher_response(text2)
                    up3 = cypher2.upper()
                    if "MATCH" in up3 and "LIMIT" in up3:
                        cypher2 = re.sub(r"LIMIT\s+(\d+)", lambda m: f"LIMIT {min(50, int(m.group(1)))}", cypher2, flags=re.IGNORECASE)
                        cypher = _clean_cypher_response(cypher2)
                        up2 = cypher.upper()
                        has_label_specific = re.search(r"MATCH\s*\(\s*\w+\s*:\s*(CLASS|FUNCTION|METHOD|FILE|MODULE)\b", up2) is not None
                        has_narrow = any(k in up2 for k in [" STARTS WITH ", " EXTENSION ", "QUALIFIED_NAME", " DECORATORS ", " CONTAINS "]) or re.search(r"-\s*\[:\w+\]\s*->", up2) is not None
                except Exception:
                    pass

            if not (has_label_specific and has_narrow):
                raise LLMGenerationError("Generated Cypher lacks required narrowing or label-specific MATCH")

            logger.info(f"[CypherGenerator] Selected Cypher: {cypher}")
            return cypher
        except Exception as e:
            logger.error(f"[CypherGenerator] Error: {e}")
            raise LLMGenerationError(f"Cypher generation failed: {e}") from e


def create_rag_orchestrator(tools: list[Tool]) -> Agent:
    """Factory function to create the main RAG orchestrator agent."""
    try:
        model_settings = None

        # Get active orchestrator model and detect its provider
        orchestrator_model_id = settings.active_orchestrator_model
        orchestrator_provider = detect_provider_from_model(orchestrator_model_id)

        if orchestrator_provider == "gemini":
            if settings.GEMINI_PROVIDER == "vertex":
                provider = GoogleVertexProvider(
                    project_id=settings.GCP_PROJECT_ID,
                    region=cast(VertexAiRegion, settings.GCP_REGION),
                    service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
                )
            else:
                provider = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)  # type: ignore

            if settings.GEMINI_THINKING_BUDGET is not None:
                model_settings = GeminiModelSettings(
                    gemini_thinking_config={
                        "thinking_budget": int(settings.GEMINI_THINKING_BUDGET)
                    }
                )

            gen_cfg = settings.get_orchestrator_generation_config()
            llm = GeminiModel(
                orchestrator_model_id,
                provider=provider,
                temperature=gen_cfg.get("temperature", 0.2),
                top_p=gen_cfg.get("top_p", 1.0),
                max_output_tokens=int(gen_cfg.get("max_tokens", 2000)),
            )
        elif orchestrator_provider == "local":
            # Local OpenAI-compatible models (Ollama) do not accept these gen args in ctor
            llm = OpenAIModel(  # type: ignore
                orchestrator_model_id,
                provider=OpenAIProvider(
                    api_key=settings.LOCAL_MODEL_API_KEY,
                    base_url=str(settings.LOCAL_MODEL_ENDPOINT),
                ),
            )
        else:  # openai provider
            # OpenAIResponsesModel does not accept temperature/top_p/max_tokens in ctor
            llm = OpenAIResponsesModel(
                orchestrator_model_id,
                provider=OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY,
                ),
            )

        return Agent(
            model=llm,
            system_prompt=RAG_ORCHESTRATOR_SYSTEM_PROMPT,
            tools=tools,
            model_settings=model_settings,
            output_retries=settings.LLM_OUTPUT_RETRIES,
        )  # type: ignore
    except Exception as e:
        raise LLMGenerationError(f"Failed to initialize RAG Orchestrator: {e}") from e
