"""
Interface Streamlit pour le TP « Prompt Engineering for Multi-Agent Systems ».

Lancez avec : streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

import main as core


@st.cache_data(show_spinner="Chargement IMDB…")
def load_imdb_cached() -> pd.DataFrame:
    return core.load_imdb_dataframe()


def main_ui() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="Prompt Engineering — IMDB",
        page_icon="🤖",
        layout="wide",
    )

    st.title("Prompt Engineering — systèmes multi-agents")
    st.caption(
        "Tokenizer (tiktoken), Groq via LangChain, jeu IMDB, prompts zero-shot / few-shot / CoT, F1 micro."
    )

    # ------------------------------------------------------------------ Sidebar
    with st.sidebar:
        st.header("Paramètres")
        eval_n = st.number_input(
            "Nombre d’avis pour l’évaluation F1",
            min_value=1,
            max_value=500,
            value=50,
            help="Chaque avis = un appel API. Réduire pour tester plus vite.",
        )
        few_n = st.number_input(
            "Exemples few-shot",
            min_value=1,
            max_value=10,
            value=core.FEW_SHOT_N,
        )
        st.divider()
        st.markdown(
            "Clé API : variable **`GROQ_API_KEY`** dans `.env` (voir README). "
            "Ne la collez pas dans l’interface."
        )

    # ------------------------------------------------------------------ Tabs
    tab_tok, tab_llm, tab_data, tab_eval, tab_class = st.tabs(
        [
            "Tokenizer",
            "LLM (Groq)",
            "Données IMDB",
            "Évaluation F1",
            "Classifier un avis",
        ]
    )

    # --- Tokenizer
    with tab_tok:
        st.subheader("Comptage de tokens (tiktoken)")
        st.text_area("Prompt système utilisé pour la classification", core.SYSTEM_PROMPT, height=120, disabled=True)
        n_tok = core.count_tokens(core.SYSTEM_PROMPT)
        st.metric("Nombre de tokens (cl100k_base)", n_tok)

    # --- LLM demo
    with tab_llm:
        st.subheader("Interaction LLM — ChatOpenAI → Groq")
        q = st.text_input("Question", value="What is an AI Agent?")
        if st.button("Envoyer au modèle", type="primary", key="btn_llm"):
            try:
                llm = core.get_llm()
                with st.spinner(f"Modèle `{core.CHAT_MODEL}`…"):
                    resp = llm.invoke([HumanMessage(content=q)])
                    text = getattr(resp, "content", str(resp))
                st.success("Réponse")
                st.write(text.strip() if isinstance(text, str) else text)
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.exception(e)

    # --- Data
    with tab_data:
        st.subheader("Jeu IMDB (datasets → pandas)")
        if st.button("Charger IMDB", key="btn_data"):
            st.session_state["imdb_loaded"] = True
        if st.session_state.get("imdb_loaded"):
            with st.spinner("Téléchargement / conversion…"):
                df = load_imdb_cached()
            ex_df, gold_df = core.split_examples_and_gold(df)
            st.session_state["examples_df"] = ex_df
            st.session_state["gold_df"] = gold_df
            st.success(
                f"Train IMDB : **{len(df)}** lignes — split : **{len(ex_df)}** exemples (pool) / **{len(gold_df)}** gold."
            )
            st.dataframe(df.head(8), use_container_width=True)
        else:
            st.info('Cliquez sur « Charger IMDB » pour afficher un aperçu.')

    # --- Evaluation
    with tab_eval:
        st.subheader("Évaluation — F1 micro (sklearn)")
        st.markdown(
            "Compare les prédictions du modèle aux étiquettes **gold** pour trois prompts : "
            "**zero-shot**, **few-shot**, **chain-of-thought**."
        )
        if st.button("Lancer l’évaluation complète", type="primary", key="btn_eval"):
            try:
                llm = core.get_llm()
            except RuntimeError as e:
                st.error(str(e))
                return

            try:
                with st.spinner("Chargement IMDB et préparation du split…"):
                    df = load_imdb_cached()
                    examples_df, gold_df = core.split_examples_and_gold(df)
                    few_shot = core.create_examples(examples_df, int(few_n))

                def predict_zero(llm_, review: str) -> str:
                    return core.invoke_sentiment(
                        llm_, core.build_zero_shot_user_prompt(review)
                    )

                def predict_few(llm_, review: str) -> str:
                    return core.invoke_sentiment(
                        llm_, core.build_few_shot_user_prompt(review, few_shot)
                    )

                def predict_cot(llm_, review: str) -> str:
                    return core.invoke_sentiment(llm_, core.build_cot_user_prompt(review))

                results: dict[str, float] = {}
                log_lines: list[str] = []

                def log_line(msg: str) -> None:
                    log_lines.append(msg)

                total_calls = int(eval_n) * 3
                bar = st.progress(0)
                status = st.empty()
                call_i = 0

                def make_on_row(name: str):
                    def _on_row(cur: int, tot: int) -> None:
                        nonlocal call_i
                        call_i += 1
                        bar.progress(min(call_i / max(total_calls, 1), 1.0))
                        status.caption(f"{name} — avis {cur}/{tot}")

                    return _on_row

                for label, pred_fn in (
                    ("Zero-shot", predict_zero),
                    ("Few-shot", predict_few),
                    ("CoT", predict_cot),
                ):
                    score = core.evaluate_prompt(
                        llm,
                        gold_df,
                        label,
                        pred_fn,
                        max_samples=int(eval_n),
                        log=log_line,
                        on_row=make_on_row(label),
                    )
                    results[label] = score

                bar.progress(1.0)
                status.empty()
                st.success("Évaluation terminée.")
                st.table(
                    pd.DataFrame(
                        [{"Prompt": k, "F1 micro": round(v, 4)} for k, v in results.items()]
                    )
                )
                with st.expander("Journal"):
                    st.code("\n".join(log_lines))
            except Exception as e:
                st.exception(e)

    # --- Single review classifier
    with tab_class:
        st.subheader("Tester un prompt sur un seul avis")
        review = st.text_area(
            "Texte de la critique",
            height=150,
            placeholder="Collez ici un avis en anglais…",
        )
        mode = st.radio(
            "Type de prompt",
            ("zero-shot", "few-shot", "cot"),
            horizontal=True,
        )
        if st.button("Classifier", key="btn_one"):
            if not review.strip():
                st.warning("Entrez un texte.")
                return
            try:
                llm = core.get_llm()
            except RuntimeError as e:
                st.error(str(e))
                return
            with st.spinner("Appel API…"):
                df = load_imdb_cached()
                examples_df, _ = core.split_examples_and_gold(df)
                few_shot = core.create_examples(examples_df, int(few_n))
                if mode == "zero-shot":
                    user = core.build_zero_shot_user_prompt(review)
                elif mode == "few-shot":
                    user = core.build_few_shot_user_prompt(review, few_shot)
                else:
                    user = core.build_cot_user_prompt(review)
                pred = core.invoke_sentiment(llm, user)
            st.metric("Sentiment prédit", pred)


if __name__ == "__main__":
    main_ui()
