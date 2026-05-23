"""
ForensicsAgent.py
-----------------
Autonomous System Log Forensics Agent for Heliox OS.

Architecture
~~~~~~~~~~~~
The agent implements a **ReAct** (Reason → Act → Observe) loop:

  1. INGEST  – load raw log lines from file, string, or list.
  2. PARSE   – `parse_syslog` → structured records.
  3. ENRICH  – `extract_timestamps` → normalised time-series.
  4. SCAN    – `scan_auth_anomalies` + `scan_system_anomalies` → findings.
  5. SCORE   – `compute_risk_score` → severity label.
  6. REASON  – LLM (Level-1 SOC Analyst prompt) analyses findings and
               produces a structured incident report.
  7. REPORT  – `AgentResult` with the JSON incident report as payload.

The LLM step is *optional*: if no API key / client is provided the agent
still returns a deterministic report from the rule-based tool outputs.

Usage
~~~~~
    from ForensicsAgent import ForensicsAgent

    agent = ForensicsAgent()

    # From a file path
    result = agent.run(log_source="/var/log/auth.log")

    # From a raw string
    result = agent.run(log_source=raw_log_text)

    # From a list of lines
    result = agent.run(log_source=lines_list)

    print(result.payload)   # structured incident report dict
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from base_agent import AgentResult, BaseAgent
from tools import (
    compute_risk_score,
    extract_timestamps,
    parse_syslog,
    scan_auth_anomalies,
    scan_system_anomalies,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SOC Analyst system prompt
# ---------------------------------------------------------------------------

_SOC_SYSTEM_PROMPT = textwrap.dedent("""
    You are a Level-1 SOC (Security Operations Center) Analyst embedded inside
    the Heliox autonomous operating system.  Your job is to analyse structured
    log findings produced by automated scanning tools and produce a concise,
    actionable incident report.

    STRICT OUTPUT CONTRACT
    ----------------------
    You MUST respond with ONLY a valid JSON object – no markdown fences,
    no preamble, no trailing commentary.  The JSON must conform exactly to
    this schema:

    {
      "incident_id":        "<string>  – unique ID, format INC-YYYYMMDD-NNNN",
      "severity":           "<CRITICAL|HIGH|MEDIUM|LOW|NONE>",
      "summary":            "<string>  – ≤3 sentence plain-English summary",
      "threat_categories":  ["<string>", ...],
      "affected_assets":    ["<string>", ...],
      "timeline": [
        {"ts": "<ISO-8601>", "event": "<string>"}
      ],
      "findings": {
        "authentication":   {"anomalies": [...], "risk": "<string>"},
        "system_integrity": {"anomalies": [...], "risk": "<string>"},
        "network":          {"anomalies": [...], "risk": "<string>"}
      },
      "recommended_actions": ["<string>", ...],
      "false_positive_notes": "<string or null>",
      "analyst_confidence":  "<HIGH|MEDIUM|LOW>"
    }

    BEHAVIOURAL RULES
    -----------------
    * Base every claim strictly on the tool findings supplied – do NOT
      invent IPs, usernames, timestamps, or processes.
    * If a category has no findings, set its anomalies list to [] and risk
      to "none detected".
    * recommended_actions must be concrete and specific (e.g.
      "Block IP 1.2.3.4 at the firewall immediately").
    * analyst_confidence reflects how clearly the evidence supports the
      severity verdict:
        HIGH   – strong corroborating signals across multiple finding types
        MEDIUM – partial or ambiguous signals
        LOW    – single weak signal or mostly unparsed logs
""").strip()


# ---------------------------------------------------------------------------
# ReAct step registry
# ---------------------------------------------------------------------------

class _ReActStep:
    """Thin descriptor for one step in the reasoning loop."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    def __repr__(self) -> str:
        return f"<ReActStep {self.name}>"


_STEPS = [
    _ReActStep("INGEST",  "Load and normalise raw log source into lines"),
    _ReActStep("PARSE",   "Identify log format and extract structured fields"),
    _ReActStep("ENRICH",  "Normalise timestamps; build time-density histogram"),
    _ReActStep("SCAN",    "Run auth + system anomaly scanners"),
    _ReActStep("SCORE",   "Compute numeric risk score and severity label"),
    _ReActStep("REASON",  "LLM SOC analyst produces structured incident report"),
    _ReActStep("REPORT",  "Assemble and return final AgentResult"),
]


# ---------------------------------------------------------------------------
# ForensicsAgent
# ---------------------------------------------------------------------------

class ForensicsAgent(BaseAgent):
    """
    Autonomous forensics agent that ingests OS logs, detects anomalies via
    a ReAct pipeline, and emits a structured incident report.

    Parameters
    ----------
    config : dict, optional
        Supported keys:
          llm_client   – an object with a compatible `.messages.create()` or
                         `openai`-style `.chat.completions.create()` interface.
                         If absent the agent runs in deterministic-only mode.
          llm_model    – model string passed to the client (default:
                         "claude-sonnet-4-20250514").
          max_log_lines – int, hard cap on lines ingested (default: 50_000).
          min_lines_for_llm – int, skip LLM if fewer lines than this
                              (avoids wasting tokens on trivial logs,
                              default: 1).
    """

    NAME = "ForensicsAgent"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        self._llm_client = self.config.get("llm_client")
        self._llm_model = self.config.get("llm_model", "claude-sonnet-4-20250514")
        self._max_lines = int(self.config.get("max_log_lines", 50_000))
        self._min_lines_for_llm = int(self.config.get("min_lines_for_llm", 1))
        self._trace: list[dict[str, Any]] = []   # ReAct execution trace

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(self, log_source: str | list[str] | Path | None = None, **kwargs: Any) -> AgentResult:
        """
        Execute the full ReAct forensics pipeline.

        Parameters
        ----------
        log_source : str | list[str] | Path
            * A filesystem path (str or Path) to a log file.
            * A multi-line string of raw log data.
            * A list of raw log line strings.
        """
        result = AgentResult(agent_name=self.NAME, status="success", payload={})
        self._trace = []

        try:
            # ── Step 1 : INGEST ─────────────────────────────────────────
            lines = self._step_ingest(log_source)

            # ── Step 2 : PARSE ──────────────────────────────────────────
            parse_result = self._step_parse(lines)

            # ── Step 3 : ENRICH ─────────────────────────────────────────
            enrich_result = self._step_enrich(parse_result["parsed"])

            # ── Step 4 : SCAN ───────────────────────────────────────────
            auth_findings, sys_findings = self._step_scan(enrich_result["records_with_ts"])

            # ── Step 5 : SCORE ──────────────────────────────────────────
            score_result = self._step_score(auth_findings, sys_findings)

            # ── Step 6 : REASON ─────────────────────────────────────────
            incident_report = self._step_reason(
                parse_stats=parse_result["stats"],
                time_info={
                    "first_event": enrich_result["first_event"],
                    "last_event": enrich_result["last_event"],
                    "density_histogram": enrich_result["density_histogram"],
                },
                auth_findings=auth_findings,
                sys_findings=sys_findings,
                score_result=score_result,
            )

            # ── Step 7 : REPORT ─────────────────────────────────────────
            result.payload = self._step_report(
                incident_report=incident_report,
                parse_stats=parse_result["stats"],
                score_result=score_result,
                react_trace=self._trace,
            )

        except Exception as exc:
            logger.exception("ForensicsAgent pipeline failed")
            result.finish(error=str(exc))
            result.payload = {"react_trace": self._trace}
            return result

        result.finish()
        return result

    # ------------------------------------------------------------------
    # ReAct steps (private)
    # ------------------------------------------------------------------

    def _step_ingest(self, log_source: Any) -> list[str]:
        step = _STEPS[0]
        self._log_action(step, {"source_type": type(log_source).__name__})

        if log_source is None:
            raise ValueError("log_source must be a file path, string, or list of lines.")

        if isinstance(log_source, (str, Path)):
            path = Path(log_source)
            try:
                _is_file = path.exists() and path.is_file()
            except OSError:
                # Path string too long or otherwise invalid as a filesystem path
                _is_file = False
            if _is_file:
                raw = path.read_text(errors="replace")
                lines = raw.splitlines()
                self._log_observation(step, {"resolved": "file", "path": str(path), "lines": len(lines)})
            else:
                # Treat as raw multi-line string
                lines = str(log_source).splitlines()
                self._log_observation(step, {"resolved": "raw_string", "lines": len(lines)})
        elif isinstance(log_source, list):
            lines = [str(l) for l in log_source]
            self._log_observation(step, {"resolved": "list", "lines": len(lines)})
        else:
            raise TypeError(f"Unsupported log_source type: {type(log_source)}")

        if len(lines) > self._max_lines:
            self._logger.warning(
                "Log source has %d lines; truncating to %d", len(lines), self._max_lines
            )
            lines = lines[: self._max_lines]

        return lines

    def _step_parse(self, lines: list[str]) -> dict[str, Any]:
        step = _STEPS[1]
        self._log_action(step, {"lines_to_parse": len(lines)})
        result = parse_syslog(lines)
        self._log_observation(step, result["stats"])
        return result

    def _step_enrich(self, parsed_records: list[dict]) -> dict[str, Any]:
        step = _STEPS[2]
        self._log_action(step, {"records": len(parsed_records)})
        result = extract_timestamps(parsed_records)
        self._log_observation(step, {
            "first_event": result["first_event"],
            "last_event": result["last_event"],
            "parse_errors": result["parse_errors"],
            "histogram_buckets": len(result["density_histogram"]),
        })
        return result

    def _step_scan(self, records: list[dict]) -> tuple[dict, dict]:
        step = _STEPS[3]
        self._log_action(step, {"records": len(records)})
        auth = scan_auth_anomalies(records)
        sys_ = scan_system_anomalies(records)
        self._log_observation(step, {
            "auth_failures": auth["total_auth_failures"],
            "brute_force_ips": len(auth["brute_force_suspects"]),
            "priv_esc_events": len(auth["privilege_escalation_attempts"]),
            "crash_events": sys_["total_crash_events"],
            "network_attack_events": sys_["total_network_attack_events"],
        })
        return auth, sys_

    def _step_score(self, auth: dict, sys_: dict) -> dict[str, Any]:
        step = _STEPS[4]
        self._log_action(step, {})
        result = compute_risk_score(auth, sys_)
        self._log_observation(step, result)
        return result

    def _step_reason(
        self,
        parse_stats: dict,
        time_info: dict,
        auth_findings: dict,
        sys_findings: dict,
        score_result: dict,
    ) -> dict[str, Any]:
        step = _STEPS[5]

        # Build a compact findings payload for the LLM prompt
        context = {
            "log_stats": parse_stats,
            "time_window": time_info,
            "risk_score": score_result,
            "auth_findings": auth_findings,
            "system_findings": sys_findings,
        }
        self._log_action(step, {"llm_available": self._llm_client is not None})

        if self._llm_client and parse_stats["total_lines"] >= self._min_lines_for_llm:
            report = self._call_llm(context)
        else:
            # Deterministic fallback – no LLM needed
            report = self._build_deterministic_report(context)

        self._log_observation(step, {
            "incident_id": report.get("incident_id"),
            "severity": report.get("severity"),
            "threat_categories": report.get("threat_categories", []),
        })
        return report

    def _step_report(
        self,
        incident_report: dict,
        parse_stats: dict,
        score_result: dict,
        react_trace: list,
    ) -> dict[str, Any]:
        step = _STEPS[6]
        self._log_action(step, {})
        payload = {
            "incident_report": incident_report,
            "pipeline_metadata": {
                "agent": self.NAME,
                "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
                "log_lines_processed": parse_stats["total_lines"],
                "parsed_successfully": parse_stats["parsed_count"],
                "risk_score": score_result["risk_score"],
                "severity": score_result["severity"],
            },
            "react_trace": react_trace,
        }
        self._log_observation(step, {"payload_keys": list(payload.keys())})
        return payload

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------

    def _call_llm(self, context: dict) -> dict[str, Any]:
        """
        Call the injected LLM client.

        Compatible with:
          * anthropic.Anthropic()  →  client.messages.create(...)
          * openai.OpenAI()        →  client.chat.completions.create(...)
        """
        user_message = (
            "Analyse the following log scan findings and produce the incident "
            "report JSON as instructed.\n\n"
            f"```json\n{json.dumps(context, indent=2, default=str)}\n```"
        )

        try:
            # Anthropic SDK style
            if hasattr(self._llm_client, "messages"):
                response = self._llm_client.messages.create(
                    model=self._llm_model,
                    max_tokens=1500,
                    system=_SOC_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                raw_json = response.content[0].text

            # OpenAI SDK style
            elif hasattr(self._llm_client, "chat"):
                response = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": _SOC_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=1500,
                )
                raw_json = response.choices[0].message.content

            else:
                raise TypeError("Unsupported LLM client interface.")

            # Strip accidental markdown fences
            raw_json = raw_json.strip()
            if raw_json.startswith("```"):
                raw_json = "\n".join(raw_json.splitlines()[1:])
            if raw_json.endswith("```"):
                raw_json = raw_json[: raw_json.rfind("```")]

            return json.loads(raw_json)

        except json.JSONDecodeError as exc:
            self._logger.error("LLM returned non-JSON output: %s", exc)
            return self._build_deterministic_report(context)
        except Exception as exc:
            self._logger.error("LLM call failed (%s); falling back to deterministic report", exc)
            return self._build_deterministic_report(context)

    # ------------------------------------------------------------------
    # Deterministic (no-LLM) report builder
    # ------------------------------------------------------------------

    def _build_deterministic_report(self, context: dict) -> dict[str, Any]:
        """
        Produce a fully structured incident report using only rule-based
        tool outputs, no LLM required.
        """
        auth = context["auth_findings"]
        sys_ = context["system_findings"]
        risk = context["risk_score"]
        tw = context["time_window"]

        incident_id = f"INC-{datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y%m%d')}-{abs(hash(json.dumps(risk))) % 9000 + 1000}"

        threat_categories: list[str] = []
        if auth["brute_force_suspects"]:
            threat_categories.append("Brute Force / Credential Stuffing")
        if auth["privilege_escalation_attempts"]:
            threat_categories.append("Privilege Escalation")
        if auth["invalid_user_attempts"]:
            threat_categories.append("Unauthorised Access Attempt")
        if sys_["system_crashes"]:
            threat_categories.append("System Stability / Crash")
        if sys_["network_attacks"]:
            threat_categories.append("Network Attack / Port Scan")

        affected_assets: list[str] = []
        for bf in auth["brute_force_suspects"][:5]:
            affected_assets.append(f"SSH service targeted from {bf['ip']}")
        for proc in sys_["repeated_crashes"]:
            affected_assets.append(f"Process: {proc}")

        summary_parts: list[str] = []
        if auth["total_auth_failures"]:
            summary_parts.append(
                f"{auth['total_auth_failures']} authentication failure(s) detected "
                f"from {auth['unique_source_ips']} unique source IP(s)."
            )
        if auth["brute_force_suspects"]:
            ips = ", ".join(b["ip"] for b in auth["brute_force_suspects"][:3])
            summary_parts.append(f"Potential brute-force activity from: {ips}.")
        if sys_["total_crash_events"]:
            summary_parts.append(f"{sys_['total_crash_events']} system crash/OOM event(s) recorded.")
        if not summary_parts:
            summary_parts.append("No significant anomalies detected in the provided log window.")

        # Build a minimal timeline from the earliest events per category
        timeline: list[dict] = []
        for bf in auth["brute_force_suspects"][:2]:
            if bf["events"]:
                timeline.append({"ts": bf["events"][0].get("ts", ""), "event": f"Brute-force attempt from {bf['ip']}"})
        for evt in auth["privilege_escalation_attempts"][:2]:
            timeline.append({"ts": evt.get("ts", ""), "event": "Privilege escalation attempt detected"})
        for crash in sys_["system_crashes"][:2]:
            timeline.append({"ts": crash.get("ts", ""), "event": f"System crash: {crash.get('process','unknown')}"})
        timeline.sort(key=lambda x: x.get("ts") or "")

        actions: list[str] = []
        for bf in auth["brute_force_suspects"][:5]:
            actions.append(f"Block IP {bf['ip']} at the host firewall (iptables / ufw) immediately.")
        if auth["privilege_escalation_attempts"]:
            actions.append("Audit sudoers file and review /var/log/auth.log for escalation chain.")
        if sys_["repeated_crashes"]:
            procs = ", ".join(sys_["repeated_crashes"].keys())
            actions.append(f"Investigate repeated crash of process(es): {procs}.")
        if sys_["total_network_attack_events"]:
            actions.append("Enable IDS/IPS rules for detected port-scan signatures.")
        if not actions:
            actions.append("Continue routine monitoring; no immediate action required.")

        return {
            "incident_id": incident_id,
            "severity": risk["severity"],
            "summary": " ".join(summary_parts),
            "threat_categories": threat_categories or ["None"],
            "affected_assets": affected_assets or ["No specific assets identified"],
            "timeline": timeline,
            "findings": {
                "authentication": {
                    "anomalies": auth["brute_force_suspects"][:5] + auth["privilege_escalation_attempts"][:5],
                    "risk": "high" if auth["total_auth_failures"] > 10 else "low",
                },
                "system_integrity": {
                    "anomalies": sys_["system_crashes"][:5],
                    "risk": "high" if sys_["total_crash_events"] > 0 else "none detected",
                },
                "network": {
                    "anomalies": sys_["network_attacks"][:5],
                    "risk": "high" if sys_["total_network_attack_events"] > 0 else "none detected",
                },
            },
            "recommended_actions": actions,
            "false_positive_notes": (
                "Automated tool; results should be reviewed by a human analyst "
                "before escalation.  Check for legitimate admin activity before "
                "blocking IPs."
            ),
            "analyst_confidence": "MEDIUM",
        }

    # ------------------------------------------------------------------
    # ReAct trace helpers
    # ------------------------------------------------------------------

    def _log_action(self, step: _ReActStep, details: dict) -> None:
        entry = {
            "step": step.name,
            "phase": "ACTION",
            "description": step.description,
            "details": details,
            "ts": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        }
        self._trace.append(entry)
        self._logger.debug("[ACTION] %s – %s", step.name, details)

    def _log_observation(self, step: _ReActStep, observation: dict) -> None:
        entry = {
            "step": step.name,
            "phase": "OBSERVATION",
            "observation": observation,
            "ts": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        }
        self._trace.append(entry)
        self._logger.debug("[OBSERVATION] %s – %s", step.name, observation)