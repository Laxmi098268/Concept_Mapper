import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
from dotenv import load_dotenv
import os
import pandas as pd
from PIL import Image
import io
import base64
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.set_page_config(page_title="LIDA: Smart Visualizations & AI Insights", layout="wide")

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY", "")


st.markdown(
    """ <style>
        .stApp {
            background-color: #1e1e1e; /* Dark grey background */
            color: #f0f0f0; /* Light text color */
        }
        .st-emotion-cache-r421ms { /* Sidebar background */
            background-color: #2d2d2d; /* Slightly lighter dark grey sidebar */
            color: #f0f0f0; /* Light text color in sidebar */
            padding: 20px;
            border-radius: 10px;
        }
        .st-emotion-cache-6qob1r { /* Main content padding */
            padding: 20px;
        }
        .st-emotion-cache-lrlib { /* Header text */
            color: #f0f0f0;
        }
        .st-emotion-cache-10pw536 { /* Subheader text */
            color: #d3d3d3;
            margin-top: 1em;
        }
        .st-emotion-cache-fqexal { /* Widget labels */
            color: #d3d3d3;
        }
        .st-emotion-cache-1w011k9 { /* Expander header */
            font-weight: bold;
            color: #f0f0f0;
        }
        .st-emotion-cache-16txtl3 { /* Success message */
            background-color: #388e3c; /* Darker green */
            color: #e8f5e9; /* Lighter green */
            padding: 10px;
            border-radius: 5px;
            margin-top: 5px;
        }
        .st-emotion-cache-10oheav { /* Info message */
            background-color: #1976d2; /* Darker blue */
            color: #e3f2fd; /* Lighter blue */
            padding: 10px;
            border-radius: 5px;
            margin-top: 5px;
        }
        .st-emotion-cache-f9w746 { /* Warning message */
            background-color: #ed6c02; /* Darker orange */
            color: #ffe0b2; /* Lighter orange */
            padding: 10px;
            border-radius: 5px;
            margin-top: 5px;
        }
        .st-emotion-cache-zufiql { /* Error message */
            background-color: #d32f2f; /* Darker red */
            color: #ffebee; /* Lighter red */
            padding: 10px;
            border-radius: 5px;
            margin-top: 5px;
        }
        .st-emotion-cache-8je8zh { /* Dataframe */
            border: 1px solid #424242; /* Darker border */
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
            background-color: #333333; /* Darker background */
            color: #f0f0f0;
        }
        .st-emotion-cache-1544g2r > div > div:first-child { /* Selectbox */
            color: #f0f0f0;
            background-color: #333333;
            border: 1px solid #424242;
            border-radius: 5px;
        }
        .st-emotion-cache-j7fxfc > div > div > div > div { /* Slider track */
            background-color: #5e5e5e;
        }
        .st-emotion-cache-j7fxfc > div > div > div > div[data-baseweb="slider-thumb"] { /* Slider thumb */
            background-color: #f0f0f0;
            border: 1px solid #5e5e5e;
        }
        .st-emotion-cache-fmj7if > label { /* Checkbox label */
            color: #f0f0f0;
        }
        .st-emotion-cache-1cpxqw2 { /* Text input */
            color: #f0f0f0;
            background-color: #333333;
            border-radius: 5px;
            border: 1px solid #424242;
            padding: 8px;
        }
        .st-emotion-cache-164nlkn { /* Button */
            background-color: #5cb85c; /* Keep your primary button color */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .st-emotion-cache-164nlkn:hover {
            background-color: #4cae4c;
        }
        .st-emotion-cache-1yt744g { /* Tabs */
            background-color: #333333;
            border-radius: 5px;
            padding: 5px;
        }
        .st-emotion-cache-am538 { /* Tab labels */
            color: #f0f0f0;
            font-weight: bold;
        }
        .st-emotion-cache-1y4p8pa { /* Tab content */
            padding: 15px;
            background-color: #2d2d2d;
            border-radius: 5px;
            margin-top: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            color: #f0f0f0;
        }
        .st-emotion-cache-10pw536 > p { /* Caption text */
            color: #a7a7a7;
            font-size: 0.9em;
            margin-top: 0.5em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


os.makedirs("data", exist_ok=True)


st.title("CONCEPT MAPPER")
st.markdown("Explore your data, generate summaries, and visualize insights with the help of language models.")
st.markdown("---")

if openai_key and openai_key.startswith("sk-"):
    api_status = ""
else:
    api_status = "🚨 API key not found. Please set OPENAI_API_KEY in your .env file."


with st.sidebar:
    st.expander("⚙️ API & Model Setup", expanded=False)
    st.write(api_status)
            
            
    st.markdown("---")
    with st.expander("⚙️ Text Generation Settings", expanded=False):
        selected_model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"])
        temperature = st.slider("🌡️ Creativity", 0.0, 1.0, 0.3)
        use_cache = st.checkbox("💾 Use Cache", value=True)

    st.markdown("---")
    st.subheader("💾 Dataset")
    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "🚗 Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
        {"label": "🌦️ Weather", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
    ]
    selected_dataset_label = st.selectbox("Choose Sample Dataset", [d["label"] for d in datasets])
    upload_own_data = st.checkbox("📤 Upload Your Own Dataset")
    
    
selected_dataset = next((d for d in datasets if d["label"] == selected_dataset_label), None)

if selected_dataset and selected_dataset["url"]:
    st.markdown(f"Download the **{selected_dataset_label}** dataset: [Download here]({selected_dataset['url']})")


selected_dataset = None
data = None

try:
    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader("📤 Upload CSV or JSON", type=["csv", "json"])
        if uploaded_file:
            uploaded_path = os.path.join("data", uploaded_file.name)
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
                data.to_csv(uploaded_path, index=False)
            elif uploaded_file.name.endswith(".json"):
                content = uploaded_file.getvalue().decode("utf-8")
                with open(uploaded_path, "w", encoding="utf-8") as f:
                    f.write(content)
                try:
                    data = pd.read_json(uploaded_path)
                except ValueError:
                    data = pd.DataFrame(json.loads(content))
            selected_dataset = uploaded_path
            st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")
    else:
        selected_dataset = next((d["url"] for d in datasets if d["label"] == selected_dataset_label), None)
        if selected_dataset:
            data = pd.read_csv(selected_dataset) if selected_dataset.endswith(".csv") else pd.read_json(selected_dataset)

except Exception as e:
    st.error(f"🚨 Error loading dataset: {e}")


if data is not None:
    with st.expander("🔍 Preview & Prepare Dataset", expanded=True):
        st.subheader("📊 Dataset Preview")
        st.markdown(f"**Rows:** {data.shape[0]}, **Columns:** {data.shape[1]}")
        st.dataframe(data.head(100), use_container_width=True)

        with st.sidebar:
            st.subheader("🛠️ Data Preparation")
            simplify = st.checkbox("🧹 Simplify large dataset", value=True)
            max_rows = st.slider("Max Rows", 0, 10000, 500)
            selected_cols = st.multiselect("Select Columns (optional)", data.columns.tolist())

        if selected_cols:
            data = data[selected_cols]

        if simplify and len(data) > max_rows:
            data = data.sample(n=max_rows, random_state=42)

        temp_path = os.path.join("data", "temp_processed.csv")
        data.to_csv(temp_path, index=False)
        selected_dataset = temp_path

if selected_dataset:
    with st.sidebar:
        st.subheader("📝 Summary Method")
        summary_options = [
            {"label": "llm", "description": "LLM-enhanced summary with column semantics"},
            
        ]
        selected_summary = st.selectbox("Summary Type", [o["label"] for o in summary_options])
        st.caption(next((o['description'] for o in summary_options if o['label'] == selected_summary), ''))

    st.subheader("📝 Dataset Summary")

    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(n=1, temperature=temperature, model=selected_model, use_cache=use_cache)

    with st.spinner("⏳ Generating dataset summary..."):
        summary = lida.summarize(selected_dataset, summary_method=selected_summary, textgen_config=textgen_config)

    if "dataset_description" in summary:
        st.info(summary["dataset_description"])

    if "fields" in summary:
        st.subheader("📊 Column Information")
        fields_df = pd.DataFrame([{
            "Column": f["column"],
            **{k: str(v) for k, v in f["properties"].items() if k != "samples"},
            "Samples": str(f["properties"].get("samples", [])),
        } for f in summary["fields"]])
        st.dataframe(fields_df, use_container_width=True)

    # -- Goal Generation --
    with st.sidebar:
        st.subheader("🎯 Goal Generation")
        num_goals = st.slider("Number of Questions", 1, 10, 4)
        own_goal = st.checkbox("➕ Add Custom Question")

        if "goals" not in st.session_state or st.button("✨ Regenerate Questions"):
            with st.spinner("⏳ Generating potential questions..."):
                st.session_state.goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)

    goals = st.session_state.goals
    goal_questions = [g.question for g in goals]

    if own_goal:
        user_goal = st.sidebar.text_input("✏️ Enter Your Question")
        if user_goal and user_goal not in goal_questions:
            custom_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
            goals.append(custom_goal)
            goal_questions.append(user_goal)

    st.subheader("🤔 Generated Questions")
    selected_goal_question = st.selectbox("Pick a Question", goal_questions)
    selected_goal_obj = next((g for g in goals if g.question == selected_goal_question), None)

    # -- Visualization --
    if selected_goal_obj:
        st.success(f"✅ Selected Question: {selected_goal_obj.question}")
        if selected_goal_obj.rationale:
            st.caption(f"Rationale: {selected_goal_obj.rationale}")

        with st.sidebar:
            st.subheader("📊 Visualization Settings")
            viz_library = st.selectbox("Visualization Library", ["seaborn", "matplotlib", "plotly"])
            num_visualizations = st.slider("Number of Visualizations", 1, 2, 1)

        textgen_config = TextGenerationConfig(n=num_visualizations, temperature=temperature, model=selected_model, use_cache=use_cache)

        st.subheader("📈 Visualizations")
        with st.spinner("⏳ Generating visualizations..."):
            visualizations = lida.visualize(summary=summary, goal=selected_goal_obj, textgen_config=textgen_config, library=viz_library)

        def download_button(image_data, filename, label):
            b64 = base64.b64encode(image_data).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{label}</a>'
            st.markdown(href, unsafe_allow_html=True)

        if visualizations:
            tabs = st.tabs([f"{getattr(viz, 'question', getattr(viz, 'title', f'Visualization {i + 1}'))}" for i, viz in enumerate(visualizations)])
            for i, viz in enumerate(visualizations):
                with tabs[i]:
                    title = getattr(viz, 'question', getattr(viz, 'title', f'Visualization {i + 1}'))
                    st.markdown(f"**{title}**")
                    if getattr(viz, 'description', None):
                        st.caption(viz.description)
                    if viz.raster:
                        img = Image.open(io.BytesIO(base64.b64decode(viz.raster)))
                        st.image(img, width=800)

                       
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        download_button(buf.getvalue(), f"visualization_{i+1}.png", "📥 Download Image")
                    elif viz.code:
                        try:
                            if viz_library == "matplotlib":
                                fig = eval(viz.code)
                                st.pyplot(fig)
                            elif viz_library == "seaborn":
                                fig = plt.figure()
                                eval(viz.code)
                                st.pyplot(fig)
                            elif viz_library == "plotly":
                                fig = eval(viz.code)
                                st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"Error rendering visualization code: {e}")
                            st.code(viz.code)
                    else:
                        st.warning("No visualization was generated for this question.")
        else:
            st.info("No visualizations were generated for the selected question. Try a different question or visualization library.")
