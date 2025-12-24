import json
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# PDF extract
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False


# Import agent-agent kamu
from aikrut_agents import (
    AikrutState,
    agent_jobvacancy_from_narrative,
    agent_extract_jv_rubric,
    agent_company_values_from_narrative,
    agent_extract_cv_general,
    agent_score_cv_against_rubric,
    agent_score_validation_doc,
)

# =========================
# Helpers
# =========================

def init_session():
    if "state" not in st.session_state:
        st.session_state.state = AikrutState()

    # “library” sederhana in-memory
    if "saved_jobvacancies" not in st.session_state:
        st.session_state.saved_jobvacancies: List[Dict[str, Any]] = []
    if "saved_company_values" not in st.session_state:
        st.session_state.saved_company_values: List[Dict[str, Any]] = []
    if "selected_job_index" not in st.session_state:
        st.session_state.selected_job_index: Optional[int] = None
    if "selected_value_index" not in st.session_state:
        st.session_state.selected_value_index: Optional[int] = None

    # storage upload pdf (POC)
    if "uploaded_cv_texts" not in st.session_state:
        st.session_state.uploaded_cv_texts: List[Dict[str, Any]] = []  # [{filename,text,ts}]

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def ensure_ready_for_cv_scoring(state: AikrutState) -> bool:
    return bool(state.extract_jv and state.company_value)

def compute_numeric_score(poin_obj: Dict[str, Any]) -> float:
    """
    Ambil rata-rata semua angka dari Character Traits / Requirements / Skills.
    """
    def values_from(section: Dict[str, Any]) -> List[float]:
        out = []
        for _, v in section.items():
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    ct = values_from(poin_obj.get("Character Traits", {}))
    rq = values_from(poin_obj.get("Requirements", {}))
    sk = values_from(poin_obj.get("Skills", {}))
    allv = ct + rq + sk
    return sum(allv) / len(allv) if allv else 0.0

def bucket_from_score(avg_score: float) -> str:
    """
    Rule-based bucket (silakan ubah threshold).
    """
    if avg_score >= 3.5:
        return "qualified"
    if avg_score >= 2.5:
        return "backup"
    return "unqualified"

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text dari PDF.
    Prioritas: pdfplumber (lebih bagus untuk layout), fallback PyPDF2.
    """
    file_bytes = uploaded_file.read()

    if _HAS_PDFPLUMBER:
        import io
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
        return "\n\n".join(text_parts).strip()

    if _HAS_PYPDF2:
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                text_parts.append(t)
        return "\n\n".join(text_parts).strip()

    raise RuntimeError("Tidak ada library PDF reader. Install: pip install pdfplumber atau PyPDF2")


def combine_final_score(score_cv: float, score_val: Optional[float], w_cv=0.7, w_val=0.3) -> float:
    if score_val is None:
        return score_cv
    return (w_cv * score_cv) + (w_val * score_val)

# =========================
# UI Pages
# =========================

def page_home():
    st.header("🏠 Home — Job Vacancy & Company Values")

    state: AikrutState = st.session_state.state

    col1, col2 = st.columns(2)

    # --- KOLOM 1: JOB VACANCY ---
    with col1:
        st.subheader("1) Buat Job Vacancy dari Naratif")
        job_narr = st.text_area(
            "Narasi job vacancy (HR menulis bebas):",
            value=state.job_narrative or "",
            height=180,
            placeholder="Contoh: Kami mencari ML Engineer untuk membangun model prediksi..."
        )

        if st.button("Generate Initial Draft", type="primary"):
            if not job_narr.strip():
                st.error("Narasi job vacancy masih kosong.")
            else:
                try:
                    state.job_narrative = job_narr
                    with st.spinner("AI sedang menyusun draf lowongan..."):
                        # Agent 1: Menghasilkan JSON seperti ML Engineer tadi
                        state = agent_jobvacancy_from_narrative(state)
                    st.success("Draf berhasil dibuat! Silakan tinjau dan edit di bawah.")
                except Exception as e:
                    st.exception(e)

        # FORM EDITOR (Akan muncul jika state.jobvacancy sudah terisi)
        if state.jobvacancy:
            st.divider()
            st.markdown("### 📝 Review & Edit Job Vacancy")
            
            # 1. Job Name
            curr_job_name = state.jobvacancy.get("job_name", "")
            edit_name = st.text_input("Job Name", value=curr_job_name)

            # 2. Job Desk (List to String)
            curr_jd_list = state.jobvacancy.get("jobdesk", [])
            jd_text = "\n".join(curr_jd_list)
            edit_jd = st.text_area("Job Description (Satu baris per poin)", value=jd_text, height=150)

            # 3. Requirement (Gabung Education, Experience, Project)
            req_obj = state.jobvacancy.get("requirement", {})
            combined_req_list = (
                req_obj.get("education", []) + 
                req_obj.get("experience", []) + 
                req_obj.get("project", [])
            )
            req_text = "\n".join(combined_req_list)
            edit_req = st.text_area("Requirements (Combined)", value=req_text, height=150)

            # Tombol Final: Save & Jalankan Agent ke-2
            if st.button("Save & Process Extracted JV"):
                # Update data dari input form kembali ke state
                state.jobvacancy["job_name"] = edit_name
                state.jobvacancy["jobdesk"] = [line.strip() for line in edit_jd.split("\n") if line.strip()]
                
                # Kita simpan requirement yang sudah digabung ke dalam list tunggal
                state.jobvacancy["requirement"] = {
                    "all_requirements": [line.strip() for line in edit_req.split("\n") if line.strip()]
                }

                try:
                    with st.spinner("Menghasilkan Rubrik (Extract_JV)..."):
                        # Agent 2: Mengambil data yang sudah diedit untuk membuat rubrik
                        state = agent_extract_jv_rubric(state)
                        
                        # Simpan ke Library
                        st.session_state.saved_jobvacancies.append({
                            "created_at": now_ts(),
                            "jobvacancy": state.jobvacancy,
                            "extract_jv": state.extract_jv
                        })
                    st.success("Job Vacancy dan Rubrik berhasil disimpan ke Library!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Gagal saat proses ekstraksi: {e}")

    # --- KOLOM 2: COMPANY VALUES ---
    with col2:
        st.subheader("2) Buat Company Values dari Naratif")
        val_narr = st.text_area(
            "Narasi company values:",
            value=state.company_value_narrative or "",
            height=180,
            placeholder="Contoh: Kami menjunjung integrity, ownership, customer empathy..."
        )

        if st.button("Generate Company Values"):
            if not val_narr.strip():
                st.error("Narasi company values masih kosong.")
            else:
                try:
                    state.company_value_narrative = val_narr
                    with st.spinner("Memanggil Agent: Company Values..."):
                        state = agent_company_values_from_narrative(state)
                    st.success("Company values berhasil dibuat.")
                except Exception as e:
                    st.exception(e)

        if state.company_value:
            st.markdown("**Company Values (generated):**")
            st.code(safe_json(state.company_value), language="json")

            if st.button("Save Company Values to Library"):
                st.session_state.saved_company_values.append({
                    "created_at": now_ts(),
                    "company_value": state.company_value
                })
                st.success("Company values tersimpan di library.")

    st.divider()
    st.subheader("3) Pilih dari Library (optional)")

    colA, colB = st.columns(2)

    with colA:
        jobs = st.session_state.saved_jobvacancies
        job_labels = [f"[{i}] {j['jobvacancy']['job_name']} — {j['created_at']}" for i, j in enumerate(jobs)]
        sel_job = st.selectbox("Pilih JobVacancy:", options=["(none)"] + job_labels, index=0)
        if sel_job != "(none)":
            idx = int(sel_job.split("]")[0].replace("[", ""))
            st.session_state.selected_job_index = idx
            chosen = jobs[idx]
            state.jobvacancy = chosen["jobvacancy"]
            state.extract_jv = chosen.get("extract_jv")
            st.info("JobVacancy aktif di-set dari library.")
            st.code(safe_json(state.jobvacancy), language="json")

    with colB:
        vals = st.session_state.saved_company_values
        val_labels = [f"[{i}] Values — {v['created_at']}" for i, v in enumerate(vals)]
        sel_val = st.selectbox("Pilih Company Values:", options=["(none)"] + val_labels, index=0)
        if sel_val != "(none)":
            idx = int(sel_val.split("]")[0].replace("[", ""))
            st.session_state.selected_value_index = idx
            chosen = vals[idx]
            state.company_value = chosen["company_value"]
            st.info("Company values aktif di-set dari library.")
            st.code(safe_json(state.company_value), language="json")


def page_cv_processing():
    st.header("📄 CV Processing — Upload PDF & Score per Kandidat")

    state: AikrutState = st.session_state.state

    if not ensure_ready_for_cv_scoring(state):
        st.warning("Sebelum proses CV, pastikan sudah ada Extract_JV dan Company Values di Home.")
        st.stop()

    st.subheader("1) Upload CV (PDF)")
    st.caption("POC: upload PDF → extract text → agent_extract_cv_general → scoring")

    files = st.file_uploader(
        "Upload CV PDF (bisa multiple):",
        type=["pdf"],
        accept_multiple_files=True
    )

    colU1, colU2 = st.columns([1, 1])
    with colU1:
        if st.button("Extract Text dari PDF"):
            if not files:
                st.warning("Belum ada file PDF diupload.")
            else:
                for f in files:
                    try:
                        text = extract_text_from_pdf(f)
                        st.session_state.uploaded_cv_texts.append({
                            "filename": f.name,
                            "text": text,
                            "uploaded_at": now_ts()
                        })
                    except Exception as e:
                        st.error(f"Gagal extract {f.name}: {e}")
                st.success("Selesai extract text dari PDF. Pilih salah satu untuk diproses.")

    with colU2:
        if st.button("Reset Upload List"):
            st.session_state.uploaded_cv_texts = []
            st.success("Upload list di-reset.")

    st.divider()

    uploaded = st.session_state.uploaded_cv_texts
    if not uploaded:
        st.info("Belum ada PDF yang diextract. Upload dan klik 'Extract Text dari PDF'.")
        st.stop()

    st.subheader("2) Pilih CV untuk diproses satu-per-satu")
    labels = [f"[{i}] {x['filename']} — {x['uploaded_at']}" for i, x in enumerate(uploaded)]
    sel = st.selectbox("Pilih CV:", options=labels)
    idx = int(sel.split("]")[0].replace("[", ""))
    chosen = uploaded[idx]

    st.markdown("**Preview text (awal):**")
    st.text_area("PDF extracted text (preview)", value=chosen["text"][:4000], height=180)

    candidate_name_override = st.text_input("Nama kandidat (opsional override):", value="")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Extract CV (General)"):
            try:
                with st.spinner("Memanggil Agent: Extract CV (General)..."):
                    extracted = agent_extract_cv_general(chosen["text"])
                state.extracted_cvs.append(extracted)
                state.cv_raw_texts.append(chosen["text"])
                st.success("Extracted CV tersimpan.")
                st.code(safe_json(extracted), language="json")
            except Exception as e:
                st.exception(e)

    with col2:
        if st.button("Score Last Extracted CV", type="primary"):
            if not state.extracted_cvs:
                st.warning("Belum ada extracted CV. Klik Extract dulu.")
            else:
                try:
                    extracted = state.extracted_cvs[-1]
                    auto_name = extracted["format_extracted"]["identity"]["full_name"]
                    use_name = candidate_name_override.strip() or auto_name

                    with st.spinner("Memanggil Agent: Score CV vs Extract_JV + Company Values..."):
                        poin = agent_score_cv_against_rubric(
                            candidate_name=use_name,
                            extracted_cv=extracted,
                            extract_jv=state.extract_jv,
                            company_value=state.company_value,
                        )
                    state.poin_cvs.append(poin)
                    st.success("Poin_CV tersimpan.")
                    st.code(safe_json(poin), language="json")
                except Exception as e:
                    st.exception(e)

    with col3:
        if st.button("Reset CV Data (Extracted + Poin)"):
            state.cv_raw_texts = []
            state.extracted_cvs = []
            state.poin_cvs = []
            st.success("Data CV di-reset.")

    st.divider()
    st.subheader("Ringkasan skor (yang sudah ada)")
    if state.poin_cvs:
        rows = []
        for p in state.poin_cvs:
            avg = compute_numeric_score(p)
            rows.append({
                "candidate": p.get("candidate"),
                "avg_score": round(avg, 2),
                "bucket": bucket_from_score(avg),
                "notes": (p.get("evidence_notes", "")[:90] + "…") if p.get("evidence_notes") else ""
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Belum ada scoring.")


def page_ranking():
    st.header("📊 Ranking — Auto Sort dari Nilai")

    state: AikrutState = st.session_state.state

    if not state.poin_cvs:
        st.warning("Belum ada Poin_CV. Silakan proses CV dulu.")
        st.stop()

    # RULE-BASED RANKING (no LLM)
    rows = []
    for p in state.poin_cvs:
        avg = compute_numeric_score(p)
        rows.append({
            "candidate": p.get("candidate"),
            "avg_score": round(avg, 3),
            "bucket": bucket_from_score(avg),
            "notes": (p.get("evidence_notes", "")[:140] + "…") if p.get("evidence_notes") else ""
        })

    df = pd.DataFrame(rows).sort_values("avg_score", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

    # Simpan ranking ke state (agar Validation bisa pakai)
    state.ranking = {
        "job_name": (state.jobvacancy["job_name"] if state.jobvacancy else "Unknown Job"),
        "ranking": [
            {"candidate": r["candidate"], "bucket": r["bucket"], "notes": r["notes"]}
            for _, r in df.iterrows()
        ]
    }

    st.divider()
    st.subheader("Ranking JSON (saved)")
    st.code(safe_json(state.ranking), language="json")

    st.divider()
    st.subheader("Detail Poin_CV (opsional)")
    with st.expander("Lihat semua JSON poin per kandidat"):
        for p in state.poin_cvs:
            st.markdown(f"### {p.get('candidate','(unknown)')} — avg={compute_numeric_score(p):.2f}")
            st.code(safe_json(p), language="json")


def page_validation():
    st.header("🧪 Validation — Upload/Test untuk Kandidat Lolos")

    state: AikrutState = st.session_state.state

    if not state.ranking:
        st.warning("Belum ada ranking. Buka halaman Ranking dulu.")
        st.stop()

    qualified = [r["candidate"] for r in state.ranking.get("ranking", []) if r["bucket"] == "qualified"]
    if not qualified:
        st.info("Tidak ada kandidat qualified saat ini.")
        st.stop()

    st.subheader("Pilih kandidat qualified untuk divalidasi")
    cand = st.selectbox("Kandidat:", options=qualified)
    

    validation_text = st.text_area(
        "Paste hasil psikotes + knowledge test (teks):",
        height=220,
        placeholder="Contoh: kandidat menjelaskan konsep ..., hasil psikotes menunjukkan ..."
    )

    if st.button("Score Validation Document", type="primary"):
        if not validation_text.strip():
            st.error("Dokumen validasi masih kosong.")
        else:
            try:
                with st.spinner("Memanggil Agent: Validation Scoring..."):
                    pv = agent_score_validation_doc(
                        candidate_name=cand,
                        validation_text=validation_text,
                        extract_jv=state.extract_jv
                    )
                state.validation_docs[cand] = validation_text
                state.poin_validation.append(pv)
                st.success("Poin_validation tersimpan.")
                st.code(safe_json(pv), language="json")
            except Exception as e:
                st.exception(e)

    st.divider()
    st.subheader("History Poin_validation")
    if state.poin_validation:
        for pv in state.poin_validation:
            st.markdown(f"### {pv.get('candidate','(unknown)')} — avg={compute_numeric_score(pv):.2f}")
            st.code(safe_json(pv), language="json")
    else:
        st.info("Belum ada validation scoring.")


def page_final_ranking():
    st.header("🏁 Final Ranking — Gabungan CV + Validation")

    state: AikrutState = st.session_state.state

    if not state.poin_cvs:
        st.warning("Belum ada Poin_CV.")
        st.stop()

    # Map candidate -> cv score
    cv_scores = {p.get("candidate"): compute_numeric_score(p) for p in state.poin_cvs}

    # Map candidate -> validation score (optional)
    val_scores = {pv.get("candidate"): compute_numeric_score(pv) for pv in state.poin_validation}

    st.subheader("Bobot skor")
    w_cv = st.slider("Bobot CV", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    w_val = 1.0 - w_cv
    st.caption(f"Bobot Validation otomatis: {w_val:.2f}")

    rows = []
    for cand, scv in cv_scores.items():
        sval = val_scores.get(cand)
        fscore = combine_final_score(scv, sval, w_cv=w_cv, w_val=w_val)
        rows.append({
            "candidate": cand,
            "cv_score": round(scv, 2),
            "validation_score": (round(sval, 2) if sval is not None else None),
            "final_score": round(fscore, 2),
            "status": "recommended" if fscore >= 3.5 else "review"
        })

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Export JSON")
    export_obj = {
        "job_name": (state.jobvacancy["job_name"] if state.jobvacancy else "Unknown Job"),
        "weights": {"cv": w_cv, "validation": w_val},
        "final_ranking": df.to_dict(orient="records"),
    }
    st.code(safe_json(export_obj), language="json")


# =========================
# Main
# =========================

def main():
    st.set_page_config(page_title="Scopai Project V1", layout="wide")
    init_session()

    st.sidebar.title("Scopai POC")
    menu = st.sidebar.radio(
        "Navigation",
        ["Home", "CV Processing", "Ranking", "Validation", "Final Ranking"],
        index=0
    )

    st.sidebar.divider()
    st.sidebar.caption("Tips: pastikan OPENROUTER_API_KEY sudah diset di environment.")
    st.sidebar.caption("Flow: Home → CV Processing → Ranking → Validation → Final Ranking")

    if menu == "Home":
        page_home()
    elif menu == "CV Processing":
        page_cv_processing()
        # pass
    elif menu == "Ranking":
        page_ranking()
        # pass
    elif menu == "Validation":
        page_validation()
        # pass
    elif menu == "Final Ranking":
        page_final_ranking()
        pass


if __name__ == "__main__":
    main()
