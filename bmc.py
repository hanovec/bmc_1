# Tento soubor je app.py pro nasazení na Streamlit Cloud

import streamlit as st
import google.generativeai as genai
import os
import uuid
import json
import re
import time
import textwrap

# ==============================================================================
# BLOK 1: KONFIGURACE A INICIALIZACE
# ==============================================================================

# --- Konfigurace stránky Streamlit ---
st.set_page_config(page_title="BMC Navigátor", layout="wide")

# --- Načtení API klíče ze Streamlit Secrets ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("API klíč pro Google Gemini není nastaven v 'Secrets' vaší Streamlit aplikace!")
    st.info("Přejděte do nastavení vaší aplikace na share.streamlit.io, klikněte na 'Manage app', poté 'Settings' -> 'Secrets' a vložte svůj klíč ve formátu: GOOGLE_API_KEY = \"vas_klic_sem\"")
    st.stop()

# --- Konfigurace Modelu ---
PRIORITY_MODEL_STEMS = ["gemini-2.5-flash-preview-05-20", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro"]
GENERATION_CONFIG = {"temperature": 0.7, "top_p": 0.95, "max_output_tokens": 65536}
model = None

@st.cache_resource
def load_model():
    model_name_to_use = None
    try:
        available_models = [m for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        for model_stem in PRIORITY_MODEL_STEMS:
            found_model = next((m for m in available_models if model_stem in m.name and 'vision' not in m.name.lower()), None)
            if found_model: model_name_to_use = found_model.name; break
        if not model_name_to_use: st.error("Nebyl nalezen žádný z prioritních modelů."); return None
        return genai.GenerativeModel(model_name=model_name_to_use, generation_config=GENERATION_CONFIG)
    except Exception as e:
        st.error(f"KRITICKÁ CHYBA při inicializaci modelu: {e}")
        return None

model = load_model()

# ==============================================================================
# BLOK 2: UI A HELPER FUNKCE
# ==============================================================================

def ai_box(content: str, title: str = "🤖 BMC Navigátor"):
    html_content = f"""<div style="border: 2px solid #4A90E2; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #F0F7FF; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"><p style="margin: 0; padding: 0; font-weight: bold; color: #4A90E2; font-family: sans-serif;">{title}</p><hr style="border: 0; border-top: 1px solid #D0E0F0; margin: 10px 0;"><p style="margin: 0; padding: 0; font-family: sans-serif; color: #333; line-height: 1.5;">{content}</p></div>"""
    st.markdown(html_content, unsafe_allow_html=True)

def user_response_box(response: str):
    html_content = f"""<div style="border: 2px solid #66BB6A; border-radius: 10px; padding: 15px; margin: 10px 0 10px 50px; background-color: #E8F5E9; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); text-align: left;"><p style="margin: 0; padding: 0; font-weight: bold; color: #2E7D32; font-family: sans-serif;">Odpověděli jste</p><hr style="border: 0; border-top: 1px solid #C8E6C9; margin: 10px 0;"><p style="margin: 0; padding: 0; font-family: sans-serif; color: #333; line-height: 1.5;"><i>„{response}“</i></p></div>"""
    st.markdown(html_content, unsafe_allow_html=True)

def ask_gemini_sdk_st(prompt_text: str, temperature: float = None) -> str:
    if not model: return "AI_ERROR: Model není inicializován."
    config_overrides = {}
    if temperature is not None: config_overrides['temperature'] = float(temperature)
    spinner_text = "AI přemýšlí..."
    if temperature: spinner_text += f" (teplota: {temperature})"
    with st.spinner(spinner_text):
        try:
            response = model.generate_content(prompt_text, generation_config=config_overrides)
            return response.text.strip()
        except Exception as e: st.error(f"CHYBA při volání API: {e}"); return f"AI_ERROR: Neočekávaná chyba: {type(e).__name__}."

# ==============================================================================
# BLOK 3: PROMPTY
# ==============================================================================
LLM_EXPERT_QUESTION_PLANNER = """Jste expert na strategické poradenství a mistr metodologie Business Model Canvas. Vaším úkolem je vytvořit strukturovaný a komplexní plán dotazování v **češtině**. Tento plán provede uživatele popisem jeho IT byznysu. DŮLEŽITÉ: Vezměte v úvahu **úvodní kontext**, který uživatel poskytl. Váš výstup MUSÍ být platný JSON seznam 9 objektů. Každý objekt musí mít **anglické** klíče: "key", "question", "coverage_points" a "examples". DŮLEŽITÉ POKYNY PRO FORMÁTOVÁNÍ: "coverage_points" a "examples" musí být seznamy plnohodnotných vět nebo frází. Veškerý text v hodnotách JSON MUSÍ být v **češtině**. Generujte POUZE JSON seznam."""
LLM_DEEP_ANALYSIS_PERSONA_V2 = """Jste strategický konzultant na úrovni partnera. Vaším úkolem je provést důkladnou strategickou analýzu poskytnutého Business Model Canvas (BMC) a vzít v úvahu **úvodní kontext** uživatele. Vaše analýza musí být strukturována: 1. Shrnutí pro vedení. 2. Hloubková analýza (SWOT s Nálezem, Důkazem a Dopadem). 3. Klíčové souvislosti. 4. Klíčové strategické otázky pro vedení. Buďte důkladný a profesionální. Odpovídejte v **češtině**."""
LLM_INNOVATION_LIST_GENERATOR = """Jste expert na obchodní inovace. Na základě BMC a analýzy vygenerujte stručný, číslovaný seznam **názvů** inovativních nápadů. Nápady rozdělte do kategorií: "Rychlá vítězství", "Strategické posuny" a "Experimentální nápady". Uveďte pouze názvy, žádné další detaily. Formátujte jako číslovaný seznam. Odpovídejte v **češtině**."""
LLM_INNOVATION_DETAIL_GENERATOR = """Jste expert na obchodní inovace. Nyní detailně rozpracujte **jeden konkrétní nápad** na inovaci, jehož název je uveden níže. Použijte tento striktní formát pro svou odpověď: **Název návrhu:**, **Popis:**, **Odůvodnění a napojení na analýzu:**, **Dopad na Business Model Canvas:**, **Akční první kroky (příštích 30 dní):** a **Možná rizika ke zvážení:**. Buďte maximálně detailní, konkrétní a akční. Odpovídejte v **češtině**."""

# ==============================================================================
# BLOK 4: HLAVNÍ TOK A UI APLIKACE
# ==============================================================================

# Inicializace Session State
if 'stage' not in st.session_state:
    st.session_state.stage = 'WELCOME'
    st.session_state.history = []
    st.session_state.user_context = ""
    st.session_state.question_plan = []
    st.session_state.current_question_index = 0
    st.session_state.bmc_data = {}
    st.session_state.analysis_result = ""
    st.session_state.innovation_titles = []

st.title("🤖 BMC Navigátor")
st.markdown("Váš AI byznys stratég pro analýzu a inovaci vašeho byznys modelu.")

# Vykreslení historie konverzace
for item in st.session_state.history:
    if item['role'] == 'ai_question':
        ai_box(item['content'], item['title'])
    elif item['role'] == 'user_response':
        user_response_box(item['content'])
    elif item['role'] == 'llm_output':
        st.markdown(f"### {item['title']}")
        st.markdown(item['content'], unsafe_allow_html=True)
        st.markdown("---")

# Logika řízení fází konverzace
if st.session_state.stage == 'WELCOME':
    ai_box("Vítejte! Než začneme, popište prosím vaši firmu, její současný byznys model a případný scénář, který chcete řešit (např. expanze, změna modelu).", "🚀 Vítejte")
    context = st.text_area("Váš popis:", height=150, key="context_input")
    if st.button("Potvrdit a zahájit analýzu"):
        if context:
            st.session_state.user_context = context
            st.session_state.history.append({'role': 'ai_question', 'title': '🚀 Vítejte', 'content': 'Vítejte! Než začneme...'})
            st.session_state.history.append({'role': 'user_response', 'content': context})
            st.session_state.stage = 'PLAN_GENERATION'
            st.rerun()
        else:
            st.warning("Prosím, zadejte popis vaší firmy.")

elif st.session_state.stage == 'PLAN_GENERATION':
    with st.spinner("AI analyzuje váš kontext a připravuje plán dotazování..."):
        prompt_with_context = f"{LLM_EXPERT_QUESTION_PLANNER}\n\nÚvodní popis od uživatele:\n{st.session_state.user_context}\n---"
        response_text = ask_gemini_sdk_st(prompt_with_context, temperature=0.2)
    if "AI_ERROR" not in response_text:
        try:
            st.session_state.question_plan = json.loads(response_text.strip().lstrip("```json").rstrip("```").strip())
            st.session_state.stage = 'DATA_GATHERING'
            st.rerun()
        except Exception as e: st.error(f"Nepodařilo se zpracovat plán od AI: {e}")
    else: st.error(response_text)

elif st.session_state.stage == 'DATA_GATHERING':
    q_index = st.session_state.current_question_index
    if q_index < len(st.session_state.question_plan):
        q_config = st.session_state.question_plan[q_index]
        q_title = f"Oblast {q_index+1}: {q_config.get('key', '').replace('_', ' ').title()}"
        q_text = q_config.get('question', 'Chybí text otázky.')
        
        # Zobrazení otázky (ale jen pokud ještě nebyla zobrazena v historii)
        if not st.session_state.history or st.session_state.history[-1].get('title') != q_title:
             st.session_state.history.append({'role': 'ai_question', 'title': q_title, 'content': q_text})
             st.rerun()
             
        # Zobrazení podotázek a příkladů, které nejsou v historii, jen jako nápověda
        st.markdown("**Pro komplexní odpověď zvažte:**")
        for point in q_config.get('coverage_points', []): st.markdown(f"- {point}")
        st.markdown(f"**Příklady:** {', '.join(q_config.get('examples', []))}")
        
        answer = st.text_area("Vaše odpověď:", height=200, key=f"answer_{q_index}")
        if st.button("Uložit a pokračovat", key=f"submit_{q_index}"):
            st.session_state.bmc_data[q_config['key']] = answer
            st.session_state.history.append({'role': 'user_response', 'content': answer})
            st.session_state.current_question_index += 1
            st.rerun()
    else:
        st.session_state.stage = 'ANALYSIS'
        st.rerun()

elif st.session_state.stage == 'ANALYSIS':
    ai_box("Skvěle, máme zmapovaný celý byznys model! Nyní provedu hloubkovou strategickou analýzu.", "🎉 Sběr dat dokončen")
    with st.spinner("AI provádí hloubkovou strategickou analýzu..."):
        bmc_data_string = "\n".join([f"- {key}: {value}" for key, value in st.session_state.bmc_data.items()])
        analysis_prompt = f"{LLM_DEEP_ANALYSIS_PERSONA_V2}\n\nÚvodní kontext od uživatele:\n{st.session_state.user_context}\n\nDetailní data z Business Model Canvas:\n{bmc_data_string}"
        analysis = ask_gemini_sdk_st(analysis_prompt, temperature=0.8)
    st.session_state.analysis_result = analysis
    st.session_state.history.append({'role': 'llm_output', 'title': 'Fáze 3: Strategická analýza', 'content': analysis})
    st.session_state.stage = 'SUGGESTION_LIST'
    st.rerun()

elif st.session_state.stage == 'SUGGESTION_LIST':
    ai_box("Na základě analýzy nyní vygeneruji několik směrů pro inovaci. Nejprve uvidíte jejich přehled a poté každou rozpracuji do detailu.", "💡 Fáze inovací")
    with st.spinner("AI generuje přehled inovativních nápadů..."):
        bmc_summary_str = "\n".join([f"- {k}: {v}" for k, v in st.session_state.bmc_data.items()])
        list_prompt = f"{LLM_INNOVATION_LIST_GENERATOR}\n\nKontext:\nPočáteční cíl uživatele:\n{st.session_state.user_context}\n\nBMC uživatele:\n{bmc_summary_str}\n\nShrnutí analýzy:\n{st.session_state.analysis_result}\n\nNyní vygenerujte stručný číslovaný seznam názvů inovací."
        innovation_list_str = ask_gemini_sdk_st(list_prompt, temperature=1.2)
    st.session_state.innovation_titles = re.findall(r'^\s*\d+\.\s*(.*)', innovation_list_str, re.MULTILINE)
    st.session_state.history.append({'role': 'llm_output', 'title': 'Přehled návrhů inovací', 'content': innovation_list_str})
    st.session_state.stage = 'SUGGESTION_DETAILS'
    st.rerun()

elif st.session_state.stage == 'SUGGESTION_DETAILS':
    ai_box(f"Nyní detailně rozpracuji těchto {len(st.session_state.innovation_titles)} nápadů.", "Detailní rozpracování")
    bmc_summary_str = "\n".join([f"- {k}: {v}" for k, v in st.session_state.bmc_data.items()])
    all_details = ""
    for title in st.session_state.innovation_titles:
        with st.spinner(f"Rozpracovávám nápad: '{title.strip()}'..."):
            detail_prompt = f"{LLM_INNOVATION_DETAIL_GENERATOR}\n\nNázev nápadu k rozpracování: '{title.strip()}'\n\nPodpůrný kontext:\n{st.session_state.user_context}\n\nBMC:\n{bmc_summary_str}\n\nAnalýza:\n{st.session_state.analysis_result}"
            detailed_suggestion = ask_gemini_sdk_st(detail_prompt, temperature=0.9)
        all_details += f"### Detail návrhu: {title.strip()}\n{detailed_suggestion}\n\n---\n\n"
    st.session_state.history.append({'role': 'llm_output', 'title': 'Podrobné návrhy inovací', 'content': all_details})
    st.session_state.stage = 'FINISHED'
    st.rerun()

elif st.session_state.stage == 'FINISHED':
    ai_box("Tímto končí naše interaktivní sezení. Pro zahájení nové analýzy obnovte stránku (F5).", title="🎉 Sezení dokončeno")
