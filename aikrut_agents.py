from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

import requests



# =========================
# Config
# =========================
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_URL = st.secrets["OPENROUTER_URL"]

DEFAULT_MODEL = st.secrets["DEFAULT_MODEL"]  # contoh
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 60


# =========================
# Minimal JSON Schema (lightweight validator)
# =========================

def _is_type(value: Any, t: Any) -> bool:
    """
    Support:
    - t as string: "object", "array", "string", "number", "integer", "boolean", "null"
    - t as list: ["string","null"] etc.
    """
    if isinstance(t, list):
        return any(_is_type(value, one) for one in t)

    if t == "object": return isinstance(value, dict)
    if t == "array": return isinstance(value, list)
    if t == "string": return isinstance(value, str)
    if t == "number": return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "integer": return isinstance(value, int) and not isinstance(value, bool)
    if t == "boolean": return isinstance(value, bool)
    if t == "null": return value is None
    return True


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validator sederhana (cukup untuk POC).
    Mendukung: type (including union list), required, properties, items, enum, minItems, maxItems.
    """
    errors: List[str] = []

    def _validate(node: Any, sch: Dict[str, Any], path: str):
        # --- type check (support union) ---
        if "type" in sch and not _is_type(node, sch["type"]):
            errors.append(f"{path}: expected {sch['type']}, got {type(node).__name__}")
            return

        # --- object ---
        if sch.get("type") == "object":
            req = sch.get("required", [])
            props = sch.get("properties", {})
            for k in req:
                if not isinstance(node, dict) or k not in node:
                    errors.append(f"{path}: missing required key '{k}'")
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in props:
                        _validate(v, props[k], f"{path}.{k}")

        # --- array ---
        elif sch.get("type") == "array":
            if not isinstance(node, list):
                return
            items = sch.get("items", {})
            min_items = sch.get("minItems")
            max_items = sch.get("maxItems")
            if min_items is not None and len(node) < min_items:
                errors.append(f"{path}: minItems {min_items}, got {len(node)}")
            if max_items is not None and len(node) > max_items:
                errors.append(f"{path}: maxItems {max_items}, got {len(node)}")
            for i, it in enumerate(node):
                _validate(it, items, f"{path}[{i}]")

        # --- enum ---
        if "enum" in sch:
            if node not in sch["enum"]:
                errors.append(f"{path}: value '{node}' not in enum {sch['enum']}")

    _validate(data, schema, "$")
    return (len(errors) == 0, errors)


# =========================
# OpenRouter client
# =========================

def call_openrouter_json(
    messages: List[Dict[str, str]],
    json_schema: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Meminta LLM mengembalikan JSON sesuai schema.
    Strategi POC:
    - Pakai prompt ketat agar output JSON-only.
    - Parse JSON dari assistant.
    - Validasi minimal schema, retry jika gagal.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY belum diset di environment variables.")

    system_guard = {
        "role": "system",
        "content": (
            "You are a strict JSON generator.\n"
            "Rules:\n"
            "1) Output MUST be valid JSON.\n"
            "2) Do NOT wrap in markdown.\n"
            "3) Do NOT add extra keys.\n"
            "4) Use null when unknown.\n"
        ),
    }

    schema_text = json.dumps(json_schema, ensure_ascii=False)
    schema_msg = {
        "role": "system",
        "content": f"Return JSON that matches this schema exactly:\n{schema_text}"
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [system_guard, schema_msg] + messages,
    }

    last_err: Optional[str] = None

    for attempt in range(max_retries + 1):
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                # optional recommended headers:
                "HTTP-Referer": "https://localhost",
                "X-Title": "Aikrut-POC",
            },
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            continue

        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        try:
            parsed = json.loads(content)
        except Exception as e:
            last_err = f"JSON parse error: {e}"
            # inject repair instruction then retry
            payload["messages"].append({
                "role": "system",
                "content": "Your last output was not valid JSON. Return ONLY valid JSON now."
            })
            continue

        ok, errs = validate_json_schema(parsed, json_schema)
        if ok:
            return parsed

        last_err = "Schema validation failed: " + "; ".join(errs[:8])
        payload["messages"].append({
            "role": "system",
            "content": f"Your JSON did not match the schema. Fix it. Errors: {last_err}"
        })

    raise RuntimeError(f"OpenRouter JSON call failed after retries. Last error: {last_err}")


# =========================
# Schemas (POC)
# =========================

JOBVACANCY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["job_name", "jobdesk", "requirement", "skill"],
    "properties": {
        "job_name": {"type": "string"},
        "jobdesk": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 12},
        "requirement": {
            "type": "object",
            "required": ["education", "experience", "project"],
            "properties": {
                "education": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
                "experience": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
                "project": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
            },
        },
        "skill": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 10},
    },
}

EXTRACT_JV_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["Character Traits", "Requirements", "Skills"],
    "properties": {
        "Character Traits": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 7},
        "Requirements": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 7},
        # Skills singkat (sesuai request kamu)
        "Skills": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 7},
    },
}

COMPANY_VALUE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["value"],
    "properties": {
        "value": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "required": ["name", "description"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        }
    },
}

# CV extracted GENERAL (untuk semua job)
EXTRACTED_CV_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["raw", "format_extracted"],
    "properties": {
        "raw": {"type": "string"},
        "format_extracted": {
            "type": "object",
            "required": ["identity", "headline", "total_years_experience", "education", "work_experience", "skills", "projects", "certifications", "languages", "soft_skills"],
            "properties": {
                "identity": {
                    "type": "object",
                    "required": ["full_name", "email", "phone", "location", "links"],
                    "properties": {
                        "full_name": {"type": "string"},
                        "email": {"type": ["string", "null"]},  # NOTE: our simple validator doesn't support union; keep null/string via prompt
                        "phone": {"type": ["string", "null"]},
                        "location": {"type": ["string", "null"]},
                        "links": {
                            "type": "object",
                            "required": ["linkedin", "github", "portfolio"],
                            "properties": {
                                "linkedin": {"type": ["string", "null"]},
                                "github": {"type": ["string", "null"]},
                                "portfolio": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
                "headline": {"type": "string"},
                "total_years_experience": {"type": "number"},
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["level", "major", "institution", "year"],
                        "properties": {
                            "level": {"type": "string"},
                            "major": {"type": "string"},
                            "institution": {"type": ["string", "null"]},
                            "year": {"type": ["number", "null"]},
                        },
                    },
                    "minItems": 0,
                    "maxItems": 5,
                },
                "work_experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["role", "company", "duration", "highlights"],
                        "properties": {
                            "role": {"type": "string"},
                            "company": {"type": ["string", "null"]},
                            "duration": {"type": "string"},
                            "highlights": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
                        },
                    },
                    "minItems": 0,
                    "maxItems": 10,
                },
                "skills": {
                    "type": "object",
                    "required": ["hard_skills", "tools", "domain"],
                    "properties": {
                        "hard_skills": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 30},
                        "tools": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 30},
                        "domain": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 20},
                    },
                },
                "projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "summary", "tech"],
                        "properties": {
                            "name": {"type": "string"},
                            "summary": {"type": "string"},
                            "tech": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 12},
                        },
                    },
                    "minItems": 0,
                    "maxItems": 20,
                },
                "certifications": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 20},
                "languages": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 10},
                "soft_skills": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 20},
            },
        },
    },
}

# Scoring (0-5) untuk tiap poin rubrik
POIN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["candidate", "Character Traits", "Requirements", "Skills", "evidence_notes"],
    "properties": {
        "candidate": {"type": "string"},
        "Character Traits": {"type": "object"},  # key: rubric_item, value: 0..5
        "Requirements": {"type": "object"},
        "Skills": {"type": "object"},
        "evidence_notes": {"type": "string"}
    },
}

RANKING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["job_name", "ranking"],
    "properties": {
        "job_name": {"type": "string"},
        "ranking": {
            "type": "array",
            "minItems": 1,
            "maxItems": 200,
            "items": {
                "type": "object",
                "required": ["candidate", "bucket", "notes"],
                "properties": {
                    "candidate": {"type": "string"},
                    "bucket": {"type": "string", "enum": ["qualified", "backup", "unqualified"]},
                    "notes": {"type": "string"},
                },
            },
        },
    },
}

POIN_VALIDATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["candidate", "Character Traits", "Requirements", "Skills", "validation_notes"],
    "properties": {
        "candidate": {"type": "string"},
        "Character Traits": {"type": "object"},
        "Requirements": {"type": "object"},
        "Skills": {"type": "object"},
        "validation_notes": {"type": "string"},
    },
}


# =========================
# State
# =========================

@dataclass
class AikrutState:
    job_narrative: Optional[str] = None
    company_value_narrative: Optional[str] = None
    jobvacancy: Optional[Dict[str, Any]] = None
    extract_jv: Optional[Dict[str, Any]] = None
    company_value: Optional[Dict[str, Any]] = None

    cv_raw_texts: List[str] = field(default_factory=list)
    extracted_cvs: List[Dict[str, Any]] = field(default_factory=list)

    poin_cvs: List[Dict[str, Any]] = field(default_factory=list)
    ranking: Optional[Dict[str, Any]] = None

    validation_docs: Dict[str, str] = field(default_factory=dict)  # candidate_name -> raw test text
    poin_validation: List[Dict[str, Any]] = field(default_factory=list)

    updated_at: float = 0.0


# =========================
# Agents
# =========================

def agent_jobvacancy_from_narrative(state: AikrutState, model: str = DEFAULT_MODEL) -> AikrutState:
    """
    Agent 1: HR narrative -> JobVacancy terstruktur
    """
    assert state.job_narrative, "job_narrative kosong"
    messages = [
        {"role": "user", "content": f"Buat job vacancy terstruktur dari narasi berikut:\n{state.job_narrative}\n\nGunakan bahasa Indonesia yang profesional."}
    ]
    state.jobvacancy = call_openrouter_json(messages, JOBVACANCY_SCHEMA, model=model)
    state.updated_at = time.time()
    return state


def agent_extract_jv_rubric(state: AikrutState, model: str = DEFAULT_MODEL) -> AikrutState:
    """
    Agent 2: JobVacancy -> Extract_JV (rubrik ringkas)
    """
    assert state.jobvacancy, "jobvacancy belum ada"
    messages = [
        {"role": "user", "content": "Ekstrak job vacancy ini menjadi rubrik penilaian (Character Traits, Requirements, Skills singkat):\n"
                                   + json.dumps(state.jobvacancy, ensure_ascii=False)}
    ]
    state.extract_jv = call_openrouter_json(messages, EXTRACT_JV_SCHEMA, model=model)
    state.updated_at = time.time()
    return state


def agent_company_values_from_narrative(state: AikrutState, model: str = DEFAULT_MODEL) -> AikrutState:
    """
    Agent 3: company value narrative -> Company_value (tanpa nama/deskripsi perusahaan)
    """
    assert state.company_value_narrative, "company_value_narrative kosong"
    messages = [
        {"role": "user", "content": f"Ubah narasi company values berikut menjadi daftar value (name, description) tanpa menyebut nama/desc perusahaan:\n{state.company_value_narrative}"}
    ]
    state.company_value = call_openrouter_json(messages, COMPANY_VALUE_SCHEMA, model=model)
    state.updated_at = time.time()
    return state


def agent_extract_cv_general(cv_text: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Agent 4: CV raw -> Extracted CV (GENERAL)
    """
    messages = [
        {"role": "user", "content": (
            "Ekstrak CV berikut menjadi format umum untuk berbagai pekerjaan.\n"
            "Jika data tidak ada, gunakan null (bukan string kosong).\n"
            "Ekstrak hanya dalam maximal 1800 token jangan lebih"
            f"CV:\n{cv_text}"
        )}
    ]

    # NOTE: schema union types (string|null) tidak didukung validator sederhana kita.
    # Untuk POC, kita tetap pakai schema ini sebagai guidance; jika validator gagal karena union,
    # kamu bisa sederhanakan tipe menjadi string dan pakai empty string.
    # Agar tetap jalan, kita buat schema versi 'string only' untuk field nullable.
    # cv_schema_safe = make_cv_schema_safe_string_only(EXTRACTED_CV_SCHEMA)
    return call_openrouter_json(messages, EXTRACTED_CV_SCHEMA, model=model)


def make_cv_schema_safe_string_only(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Biar validator sederhana tidak error: ganti ["string","null"] jadi "string".
    Untuk POC, null direpresentasikan sebagai string kosong "".
    """
    def _walk(s: Any) -> Any:
        if isinstance(s, dict):
            out = {}
            for k, v in s.items():
                if k == "type" and isinstance(v, list):
                    # prefer string if exists
                    if "string" in v:
                        out[k] = "string"
                    elif "number" in v:
                        out[k] = "number"
                    else:
                        out[k] = v[0]
                else:
                    out[k] = _walk(v)
            return out
        if isinstance(s, list):
            return [_walk(x) for x in s]
        return s
    return _walk(schema)


def agent_score_cv_against_rubric(
    candidate_name: str,
    extracted_cv: Dict[str, Any],
    extract_jv: Dict[str, Any],
    company_value: Dict[str, Any],
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Agent 5: Scoring CV vs rubrik Extract_JV + company values.
    Output: Poin_CV (0-5 per poin).
    """
    scoring_instructions = (
        "Skor setiap poin rubrik dengan skala 0-5:\n"
        "0=Not mentioned\n1=Weakly implied\n2=Mentioned once without detail\n"
        "3=Explicit minimal evidence\n4=Explicit multiple evidence\n5=Strong emphasized\n"
        "Berikan skor objektif hanya dari evidence di extracted_cv. Jangan mengarang.\n"
    )

    messages = [
        {"role": "user", "content": (
            f"Nama kandidat: {candidate_name}\n\n"
            f"{scoring_instructions}\n"
            "Rubrik Extract_JV:\n" + json.dumps(extract_jv, ensure_ascii=False) + "\n\n"
            "Company values:\n" + json.dumps(company_value, ensure_ascii=False) + "\n\n"
            "Extracted CV (GENERAL):\n" + json.dumps(extracted_cv, ensure_ascii=False) + "\n\n"
            "Kembalikan JSON dengan key rubrik persis sama, nilai angka 0-5.\n"
            "Tambahkan evidence_notes singkat (1 paragraf) kenapa skor demikian."
        )}
    ]

    # Untuk POC: schema longgar di bagian objek rubrik.
    return call_openrouter_json(messages, POIN_SCHEMA, model=model)


def agent_rank_candidates(
    job_name: str,
    poin_cvs: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Agent 6: Buat ranking qualified/backup/unqualified dari hasil poin.
    (Boleh pakai rule-based, tapi ini versi LLM agar cepat.)
    """
    messages = [
        {"role": "user", "content": (
            "Urutkan kandidat berdasarkan kualitas keseluruhan.\n"
            "Buat bucket: qualified, backup, unqualified.\n"
            "Beri notes singkat untuk tiap kandidat.\n"
            f"Job: {job_name}\n\n"
            "Poin kandidat:\n" + json.dumps(poin_cvs, ensure_ascii=False)
        )}
    ]
    return call_openrouter_json(messages, RANKING_SCHEMA, model=model)


def agent_score_validation_doc(
    candidate_name: str,
    validation_text: str,
    extract_jv: Dict[str, Any],
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Agent 7: Skor dokumen validasi (psikotes/knowledge test) dengan rubrik yang sama.
    """
    messages = [
        {"role": "user", "content": (
            "Gunakan rubrik yang sama untuk menilai dokumen validasi berikut.\n"
            "Berikan skor 0-5 per poin. Jangan mengarang.\n\n"
            "Rubrik Extract_JV:\n" + json.dumps(extract_jv, ensure_ascii=False) + "\n\n"
            f"Candidate: {candidate_name}\n\n"
            "Dokumen validasi:\n" + validation_text
        )}
    ]
    return call_openrouter_json(messages, POIN_VALIDATION_SCHEMA, model=model)

