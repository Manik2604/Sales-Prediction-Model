import os
import gradio as gr
import joblib
import pandas as pd

# =====================================
# Load Trained Model
# =====================================
model = joblib.load("Sales_Prediction_Model.pkl")

# =====================================
# Prediction Function
# =====================================
def predict_sales(tv, radio, newspaper):
    data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    prediction = model.predict(data)[0]

    return f"📈 Predicted Sales: {prediction:.2f} units"

# =====================================
# Custom CSS
# =====================================
custom_css = """
.gradio-container{
    background-image:url('https://images.unsplash.com/photo-1556740749-887f6717d7e4?q=80&w=2070&auto=format&fit=crop');
    background-size:cover;
    background-position:center;
    background-attachment:fixed;
}

.glass-container{
    background:rgba(255,255,255,0.95);
    border-radius:18px;
    padding:25px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.2);
    color:#222 !important;
}

.glass-container h1,
.glass-container h2,
.glass-container h3,
.glass-container p,
.glass-container li,
.glass-container strong{
    color:#222 !important;
}

footer{
    visibility:hidden;
}
"""

# =====================================
# Gradio Interface
# =====================================
with gr.Blocks(
    css=custom_css,
    title="Sales Prediction using Decision Tree"
) as interface:

    with gr.Column(elem_classes="glass-container"):

        gr.Markdown(
            """
            # 📊 Sales Prediction using Machine Learning
            ### Predict Product Sales based on Advertisement Budget
            """
        )

        gr.HTML("<hr>")

        with gr.Row():

            # Left Side
            with gr.Column(scale=2):

                gr.Markdown("## 📥 Enter Advertisement Budget")

                tv = gr.Number(
                    label="TV Advertisement Budget",
                    value=100
                )

                radio = gr.Number(
                    label="Radio Advertisement Budget",
                    value=25
                )

                newspaper = gr.Number(
                    label="Newspaper Advertisement Budget",
                    value=20
                )

                predict_btn = gr.Button(
                    "Predict Sales",
                    variant="primary"
                )

                output = gr.Textbox(
                    label="Prediction Result"
                )

                predict_btn.click(
                    fn=predict_sales,
                    inputs=[tv, radio, newspaper],
                    outputs=output
                )

            # Right Side
            with gr.Column(scale=1):

                gr.Markdown("## 👩‍💻 Developer")

                gr.Markdown("""
**Name:** Manik Jindal

**College:**
Panipat Institute of Engineering and Technology

**Project:**
Sales Prediction using Decision Tree Regression

**Technology Stack**

- Python
- Scikit-Learn
- Decision Tree Regressor
- Pandas
- Joblib
- Gradio


""")

                gr.Markdown("## 📌 About Project")

                gr.Markdown("""
This application predicts the expected product sales based on the advertising expenditure on:

- 📺 TV
- 📻 Radio
- 📰 Newspaper

The prediction is generated using a trained **Decision Tree Regression** model.
""")

# =====================================
# Launch App
# =====================================
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
