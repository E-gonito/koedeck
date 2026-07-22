"""Setup checker for koedeck.

Validates all external dependencies:
- ElevenLabs API key (from .env)
- ffmpeg availability
- Ollama/LLM connectivity and model availability
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx
from dotenv import load_dotenv


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    fix_hint: str = ""


@dataclass
class SetupStatus:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.status in (CheckStatus.PASS, CheckStatus.WARN) for c in self.checks)

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]


def check_env_file() -> CheckResult:
    env_path = Path(".env")
    if env_path.exists():
        return CheckResult(name=".env file", status=CheckStatus.PASS, message=".env file found")
    return CheckResult(
        name=".env file", status=CheckStatus.FAIL, message=".env file not found",
        fix_hint="Run: cp .env.example .env  then add your ElevenLabs API key",
    )


def check_api_key() -> CheckResult:
    load_dotenv()
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key or key == "your-key-here":
        return CheckResult(
            name="ElevenLabs API key", status=CheckStatus.FAIL,
            message="API key not configured",
            fix_hint="Edit .env and set ELEVENLABS_API_KEY=your-actual-key",
        )
    masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
    return CheckResult(name="ElevenLabs API key", status=CheckStatus.PASS, message=f"API key found ({masked})")


async def check_api_key_valid() -> CheckResult:
    load_dotenv()
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key or key == "your-key-here":
        return CheckResult(name="ElevenLabs API connectivity", status=CheckStatus.SKIP, message="Skipped - no API key configured")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.elevenlabs.io/v1/user", headers={"xi-api-key": key})
        if resp.status_code == 200:
            data = resp.json()
            tier = data.get("subscription", {}).get("tier", "unknown")
            return CheckResult(name="ElevenLabs API connectivity", status=CheckStatus.PASS, message=f"Connected - tier: {tier}")
        elif resp.status_code == 401:
            return CheckResult(name="ElevenLabs API connectivity", status=CheckStatus.FAIL, message="Invalid API key (401)", fix_hint="Check your key at https://elevenlabs.io/app/settings/api-keys")
        else:
            return CheckResult(name="ElevenLabs API connectivity", status=CheckStatus.WARN, message=f"HTTP {resp.status_code}")
    except httpx.TimeoutException:
        return CheckResult(name="ElevenLabs API connectivity", status=CheckStatus.WARN, message="Timed out - check internet")
    except Exception as e:
        return CheckResult(name="ElevenLabs API connectivity", status=CheckStatus.FAIL, message=f"Failed: {str(e)[:80]}", fix_hint="Check your internet connection")


def check_ffmpeg() -> CheckResult:
    if shutil.which("ffmpeg"):
        return CheckResult(name="ffmpeg", status=CheckStatus.PASS, message="ffmpeg found on PATH")
    return CheckResult(name="ffmpeg", status=CheckStatus.FAIL, message="ffmpeg not found", fix_hint="Install with: brew install ffmpeg")


async def check_ollama_connectivity(base_url: str = "http://localhost:11434") -> CheckResult:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return CheckResult(name="Ollama connectivity", status=CheckStatus.PASS, message=f"Connected - {len(models)} model(s)")
        return CheckResult(name="Ollama connectivity", status=CheckStatus.WARN, message=f"HTTP {resp.status_code}")
    except httpx.ConnectError:
        return CheckResult(name="Ollama connectivity", status=CheckStatus.FAIL, message="Cannot connect", fix_hint="Start Ollama: open the app, or run: ollama serve")
    except Exception as e:
        return CheckResult(name="Ollama connectivity", status=CheckStatus.FAIL, message=f"{str(e)[:80]}", fix_hint="Install from https://ollama.ai/")


async def check_ollama_model(model: str = "qwen3:8b", base_url: str = "http://localhost:11434") -> CheckResult:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
        if resp.status_code != 200:
            return CheckResult(name=f"Model: {model}", status=CheckStatus.SKIP, message="Ollama not reachable")
        models = resp.json().get("models", [])
        model_base = model.split(":")[0]
        available = [m.get("name", "") for m in models]
        if model in available or any(model_base in name for name in available):
            return CheckResult(name=f"Model: {model}", status=CheckStatus.PASS, message=f"{model} available")
        return CheckResult(name=f"Model: {model}", status=CheckStatus.FAIL, message=f"{model} not found", fix_hint=f"Pull with: ollama pull {model}")
    except Exception:
        return CheckResult(name=f"Model: {model}", status=CheckStatus.SKIP, message="Ollama not reachable")


async def run_all_checks(llm_base_url: str = "http://localhost:11434", llm_model: str = "qwen3:8b") -> SetupStatus:
    status = SetupStatus()
    status.checks.append(check_env_file())
    status.checks.append(check_api_key())
    status.checks.append(check_ffmpeg())
    status.checks.append(await check_api_key_valid())
    status.checks.append(await check_ollama_connectivity(llm_base_url))
    status.checks.append(await check_ollama_model(llm_model, llm_base_url))
    return status


def is_first_launch() -> bool:
    env_exists = Path(".env").exists()
    project_exists = Path("project.json").exists()
    return not env_exists or not project_exists
