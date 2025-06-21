# ==============================================================================
# BLOK 1: INSTALACE A KONFIGURACE
# ==============================================================================
# Instalace potřebných knihoven
!pip install -q google-generativeai
!pip install -q python-dotenv

import os
import time
import google.generativeai as genai
import json
from IPython.display import display, HTML
import textwrap
import re

# --- Konfigurace API klíče ---
try:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    print("API klíč úspěšně načten z Google Colab Secrets.")
except ImportError:
    print("Nelze načíst z Colab Secrets. Ujistěte se, že spouštíte v Google Colabu.")
    GOOGLE_API_KEY = None

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key není nastaven. V Google Colabu jej nastavte v sekci 'Tajemství' (Secrets) s názvem 'GOOGLE_API_KEY'.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google Generative AI API konfigurováno.")

# --- Konfigurace Modelu ---
PRIORITY_MODEL_STEMS = [
    "gemini-2.5-flash-preview-05-20", # VÁMI POŽADOVANÝ MODEL
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-pro",
]
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_output_tokens": 65536,
}
model = None

try:
    print("\nHledám dostupný Gemini model...")
    model_name_to_use = None
    available_models = [m for m in genai.list_models() if "generateContent" in m.supported_generation_methods]

    for model_stem in PRIORITY_MODEL_STEMS:
        found_model = next((m for m in available_models if model_stem in m.name and 'vision' not in m.name.lower()), None)
        if found_model:
            model_name_to_use = found_model.name
            print(f"  > Nalezen prioritní model: {model_name_to_use}")
            break

    if not model_name_to_use:
        raise ValueError("Nebyl nalezen žádný z prioritních modelů. Zkontrolujte dostupnost modelů ve vašem regionu.")

    model = genai.GenerativeModel(
        model_name=model_name_to_use,
        generation_config=GENERATION_CONFIG
    )
    print(f"Model '{model_name_to_use}' úspěšně inicializován.")
except Exception as e:
    print(f"KRITICKÁ CHYBA při inicializaci modelu: {e}")


# ==============================================================================
# BLOK 2: UI A HELPER FUNKCE (VŠE V ČEŠTINĚ)
# ==============================================================================

def wrap_text(text, width=110):
    return '<br>'.join(textwrap.wrap(text, width=width, replace_whitespace=False))

def ai_box(content: str, title: str = "🤖 BMC Navigátor"):
    wrapped_content = wrap_text(content)
    html_content = f"""<div style="border: 2px solid #4A90E2; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #F0F7FF; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"><p style="margin: 0; padding: 0; font-weight: bold; color: #4A90E2; font-family: sans-serif;">{title}</p><hr style="border: 0; border-top: 1px solid #D0E0F0; margin: 10px 0;"><p style="margin: 0; padding: 0; font-family: sans-serif; color: #333; line-height: 1.5;">{wrapped_content}</p></div>"""
    display(HTML(html_content))

def user_prompt_box(prompt_text: str) -> str:
    html_prompt = f"""<div style="border: 1px solid #50a14f; border-radius: 5px; padding: 10px; margin: 10px 0; background-color: #F7FFF7;"><p style="margin: 0; padding: 0; font-weight: bold; color: #387038; font-family: sans-serif;">✍️ Úkol pro vás</p><p style="margin: 0; padding: 5px 0 0 0; font-family: sans-serif; color: #333;">{prompt_text}</p></div>"""
    display(HTML(html_prompt))
    return input("Vaše odpověď > ")

def display_user_response(response: str):
    wrapped_response = wrap_text(response)
    html_content = f"""<div style="border: 2px solid #66BB6A; border-radius: 10px; padding: 15px; margin: 10px 0 10px 50px; background-color: #E8F5E9; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); text-align: left;"><p style="margin: 0; padding: 0; font-weight: bold; color: #2E7D32; font-family: sans-serif;">Odpověděli jste</p><hr style="border: 0; border-top: 1px solid #C8E6C9; margin: 10px 0;"><p style="margin: 0; padding: 0; font-family: sans-serif; color: #333; line-height: 1.5;"><i>„{wrapped_response}“</i></p></div>"""
    display(HTML(html_content))

def display_status_message(message: str):
    html_content = f"""<p style="font-family: sans-serif; color: #888; font-style: italic; text-align: center; margin: 10px 0;">{message}</p>"""
    display(HTML(html_content))

def display_llm_output(title: str, content: str):
    from IPython.display import Markdown
    print(f"\n--- {title} ---")
    if "AI_ERROR" in content:
        ai_box(content, title="❌ Nastala chyba")
    else:
        display(Markdown(content))
    print("--------------------------------------------------")

def ask_gemini_sdk(prompt_text: str, temperature: float = None) -> str:
    if not model: return "AI_ERROR: Model není inicializován."
    config_overrides = {}
    if temperature is not None:
        config_overrides['temperature'] = float(temperature)
        display_status_message(f"AI přemýšlí s teplotou: {config_overrides['temperature']}...")
    else:
        display_status_message("AI přemýšlí s výchozí teplotou...")
    try:
        response = model.generate_content(prompt_text, generation_config=config_overrides)
        if response.parts: return response.text.strip()
        elif response.prompt_feedback: return f"AI_ERROR: Požadavek byl zablokován ({response.prompt_feedback.block_reason.name})."
        else: return "AI_ERROR: Obdržena nekompletní odpověď od modelu."
    except Exception as e:
        display_status_message(f"CHYBA při volání API: {e}")
        return f"AI_ERROR: Neočekávaná chyba: {type(e).__name__}."

# ==============================================================================
# BLOK 3: PROMPTY A LOGIKA APLIKACE
# ==============================================================================

LLM_EXPERT_QUESTION_PLANNER = """Jste expert na strategické poradenství a mistr metodologie Business Model Canvas. Vaším úkolem je vytvořit strukturovaný a komplexní plán dotazování v **češtině**. Tento plán provede uživatele popisem jeho IT byznysu. DŮLEŽITÉ: Vezměte v úvahu **úvodní kontext**, který uživatel poskytl. Váš výstup MUSÍ být platný JSON seznam 9 objektů. Každý objekt musí mít **anglické** klíče: "key", "question", "coverage_points" a "examples". DŮLEŽITÉ POKYNY PRO FORMÁTOVÁNÍ: "coverage_points" a "examples" musí být seznamy plnohodnotných vět nebo frází. Veškerý text v hodnotách JSON MUSÍ být v **češtině**. Generujte POUZE JSON seznam."""
LLM_DEEP_ANALYSIS_PERSONA_V2 = """Jste strategický konzultant na úrovni partnera. Vaším úkolem je provést důkladnou strategickou analýzu poskytnutého Business Model Canvas (BMC) a vzít v úvahu **úvodní kontext** uživatele. Vaše analýza musí být strukturována: 1. Shrnutí pro vedení. 2. Hloubková analýza (SWOT s Nálezem, Důkazem a Dopadem). 3. Klíčové souvislosti. 4. Klíčové strategické otázky pro vedení. Buďte důkladný a profesionální. Odpovídejte v **češtině**."""
LLM_INNOVATION_LIST_GENERATOR = """Jste expert na obchodní inovace. Na základě BMC a analýzy vygenerujte stručný, číslovaný seznam **názvů** inovativních nápadů. Nápady rozdělte do kategorií: "Rychlá vítězství", "Strategické posuny" a "Experimentální nápady". Uveďte pouze názvy, žádné další detaily. Formátujte jako číslovaný seznam. Odpovídejte v **češtině**."""
LLM_INNOVATION_DETAIL_GENERATOR = """Jste expert na obchodní inovace. Nyní detailně rozpracujte **jeden konkrétní nápad** na inovaci, jehož název je uveden níže. Použijte tento striktní formát pro svou odpověď: **Název návrhu:**, **Popis:**, **Odůvodnění a napojení na analýzu:**, **Dopad na Business Model Canvas:**, **Akční první kroky (příštích 30 dní):** a **Možná rizika ke zvážení:**. Buďte maximálně detailní, konkrétní a akční. Odpovídejte v **češtině**."""

def get_initial_user_context() -> str:
    prompt_text = "Než začneme, popište prosím volným textem vaši firmu, její současný byznys model a případný scénář, který chcete řešit (např. expanze, změna modelu). Čím více kontextu mi dáte, tím relevantnější bude naše další práce."
    user_response = user_prompt_box(prompt_text)
    display_user_response(user_response)
    ai_box("Děkuji za kontext. Nyní připravím plán dotazování na míru vaší situaci.", title="✅ Kontext přijat")
    return user_response

def get_user_input_with_llm_validation(bmc_block_question: str, block_name: str, coverage_points: list = None, it_examples: list[str] = None) -> str:
    full_prompt_html = f"<p style='font-size: 1.1em; margin-bottom: 15px;'>{bmc_block_question}</p>"
    if coverage_points:
        full_prompt_html += "<p style='margin-bottom: 5px; font-weight: bold;'>Pro komplexní odpověď zvažte prosím následující body:</p><ul style='margin-top: 5px; margin-left: 20px;'>"
        for point in coverage_points: full_prompt_html += f"<li style='margin-bottom: 5px;'>{point}</li>"
        full_prompt_html += "</ul>"
    if it_examples:
        full_prompt_html += f"<p style='margin-top: 15px; font-style: italic; color: #555;'>Například: {', '.join(it_examples)}.</p>"
    user_response = user_prompt_box(full_prompt_html)
    user_response_stripped = user_response.strip()
    display_user_response(user_response_stripped)
    if user_response_stripped.lower() in ["n/a", "ne", "přeskočit", "skip"]:
        ai_box(f"Rozumím. Přeskočíme oblast '{block_name}'.", title="✅ Potvrzeno")
        return "Skipped"
    return user_response_stripped

def generate_question_plan(user_context: str) -> list:
    ai_box("Na základě vašeho popisu připravuji personalizovaný plán dotazování...", title="🧠 Příprava plánu")
    prompt_with_context = f"{LLM_EXPERT_QUESTION_PLANNER}\n\nÚvodní popis od uživatele (použijte jako kontext):\n---\n{user_context}\n---"
    response_text = ask_gemini_sdk(prompt_with_context, temperature=0.2)
    if "AI_ERROR" in response_text:
        ai_box("Nepodařilo se mi vytvořit plán. Zkuste prosím spustit buňku znovu.", title="❌ Chyba plánu")
        return []
    try:
        cleaned_json_text = response_text.strip().lstrip("```json").rstrip("```").strip()
        question_plan = json.loads(cleaned_json_text)
        if isinstance(question_plan, list) and all('key' in item for item in question_plan):
            ai_box(f"Plán dotazování byl úspěšně vygenerován. Zeptám se vás na {len(question_plan)} klíčových oblastí.", title="✅ Plán připraven")
            return question_plan
        else: raise ValueError("Vygenerovaný JSON postrádá požadované klíče.")
    except (json.JSONDecodeError, ValueError) as e:
        ai_box(f"Nastala chyba při zpracování vygenerovaného plánu: {e}.", title="❌ Chyba zpracování")
        return []

def conduct_dynamic_bmc_analysis(question_plan: list) -> dict:
    ai_box("Nyní společně projdeme jednotlivé bloky vašeho byznys modelu do hloubky.", title="🚀 Jdeme na to")
    bmc_data = {}
    for i, config in enumerate(question_plan):
        display_status_message(f"Oblast {i+1} z {len(question_plan)}: {config.get('key', 'Neznámý blok').replace('_', ' ').title()}")
        response = get_user_input_with_llm_validation(
            bmc_block_question=config.get('question', 'Chybí text otázky.'),
            block_name=config.get('key', f'Otázka {i+1}'),
            coverage_points=config.get('coverage_points', []),
            it_examples=config.get('examples', [])
        )
        bmc_data[config.get('key', f'custom_question_{i+1}')] = response
    ai_box("Skvělá práce! Zmapovali jsme celý váš byznys model.", title="🎉 Hotovo")
    return bmc_data

def perform_llm_bmc_analysis(bmc_data: dict, user_context: str) -> str:
    display_status_message("Zahajuji expertní strategickou analýzu...")
    bmc_data_string = "\n".join([f"- {key}: {value}" for key, value in bmc_data.items() if value != "Skipped"])
    analysis_prompt = f"{LLM_DEEP_ANALYSIS_PERSONA_V2}\n\nÚvodní kontext od uživatele:\n{user_context}\n\nDetailní data z Business Model Canvas:\n{bmc_data_string}"
    return ask_gemini_sdk(analysis_prompt, temperature=0.8)

def generate_innovation_list(bmc_data_str: str, analysis_result: str, user_context: str) -> str:
    display_status_message("Generuji přehled inovativních nápadů...")
    list_prompt = f"{LLM_INNOVATION_LIST_GENERATOR}\n\nKontext:\nPočáteční cíl uživatele:\n{user_context}\n\nBMC uživatele:\n{bmc_data_str}\n\nShrnutí analýzy:\n{analysis_result}\n\nNyní vygenerujte stručný číslovaný seznam názvů inovací."
    return ask_gemini_sdk(list_prompt, temperature=1.2)

def generate_detailed_innovation(innovation_title: str, bmc_data_str: str, analysis_result: str, user_context: str) -> str:
    display_status_message(f"Rozpracovávám detailně nápad: '{innovation_title}'...")
    detail_prompt = f"{LLM_INNOVATION_DETAIL_GENERATOR}\n\nNázev nápadu k rozpracování: '{innovation_title}'\n\nPodpůrný kontext:\nPočáteční cíl uživatele:\n{user_context}\n\nBMC uživatele:\n{bmc_data_str}\n\nShrnutí analýzy:\n{analysis_result}\n\nNyní detailně rozpracujte tento jeden nápad podle zadaného formátu."
    return ask_gemini_sdk(detail_prompt, temperature=0.9)

# ==============================================================================
# BLOK 4: HLAVNÍ SPUŠTĚCÍ FUNKCE
# ==============================================================================

def run_main_session():
    """Orchestruje celé sezení s BMC Navigátorem."""
    if not model:
        ai_box("Gemini model nebyl inicializován. Sezení nemůže začít.", title="Kritická chyba")
        return

    # Fáze 1: Uvítání a získání kontextu
    ai_box("Vítejte v BMC Navigátoru! Jsem váš AI byznys stratég připravený pomoci vám analyzovat a inovovat váš byznys model.", title="🚀 Vítejte")
    user_context = get_initial_user_context()
    if not user_context:
        ai_box("Bez úvodního popisu nemůžeme pokračovat. Zkuste prosím spustit sezení znovu.", title="❌ Chybí kontext")
        return

    # Fáze 2: Dotazování
    question_plan = generate_question_plan(user_context)
    if not question_plan:
        ai_box("Nepodařilo se mi připravit plán dotazování. Zkuste prosím spustit sezení znovu.", title="❌ Chyba spuštění")
        return
    current_bmc_data = conduct_dynamic_bmc_analysis(question_plan)

    # Fáze 3: Analýza
    analysis_result = perform_llm_bmc_analysis(current_bmc_data, user_context)
    display_llm_output("Fáze 3: Strategická analýza", analysis_result)

    # Fáze 4: Inovace (dvoufázová)
    ai_box("Na základě analýzy nyní vygeneruji několik směrů pro inovaci vašeho byznys modelu. Nejprve uvidíte jejich přehled a poté každou rozpracuji do detailu.", title="💡 Fáze inovací")
    bmc_summary_str = "\n".join([f"- {k}: {v}" for k, v in current_bmc_data.items() if v != "Skipped"])

    # Krok 4a: Získání seznamu inovací
    innovation_list_str = generate_innovation_list(bmc_summary_str, analysis_result, user_context)
    display_llm_output("Přehled návrhů inovací", innovation_list_str)

    # Krok 4b: Rozpracování každé inovace
    innovation_titles = re.findall(r'^\s*\d+\.\s*(.*)', innovation_list_str, re.MULTILINE)
    if not innovation_titles:
        ai_box("Nepodařilo se mi extrahovat názvy inovací ze seznamu. Pokračování není možné.", title="Chyba zpracování")
        return

    ai_box(f"Nyní detailně rozpracuji těchto {len(innovation_titles)} nápadů.", title="Detailní rozpracování")
    for title in innovation_titles:
        # OPRAVENO: Použita správná proměnná 'analysis_result' místo neexistující 'analysis_summary'
        detailed_suggestion = generate_detailed_innovation(title.strip(), bmc_summary_str, analysis_result, user_context)
        display_llm_output(f"Detail návrhu: {title.strip()}", detailed_suggestion)
        time.sleep(1)

    # Závěr
    ai_box("Tímto končí naše interaktivní sezení. Doufám, že detailní analýza a návrhy byly přínosné pro vaše strategické plánování.", title="🎉 Sezení dokončeno")

# --- Spuštění celé aplikace ---
if __name__ == "__main__":
    run_main_session()
